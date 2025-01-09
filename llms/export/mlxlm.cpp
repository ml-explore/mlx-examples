// Copyright © 2024 Apple Inc.

#include <chrono>
#include <iomanip>
#include <iostream>

#include "mlxlm.h"

namespace mx = mlx::core;

#define seconds(x)                                                             \
  (std::chrono::duration_cast<std::chrono::nanoseconds>(x).count() / 1e9)
#define time_now() std::chrono::high_resolution_clock::now()

// Maybe compile
std::function<mx::Args(mx::Args)> load_model(const std::string &path) {
  return mx::compile(mx::import_function(path), /* shapeless = */ true);
}

// Maybe make tokenizer virtual
BPETokenizer load_tokenizer(const std::string &path) {
  return BPETokenizer(path);
}

void generate(const std::function<mx::Args(mx::Args)> &model,
              const BPETokenizer &tokenizer, const std::string &prompt,
              int max_tokens /* = 256 */) {

  auto prompt_tokens = tokenizer.encode(prompt);
  int prompt_size = prompt_tokens.size();
  auto y = mx::array(prompt_tokens.data(), {1, prompt_size}, mx::uint32);

  auto create_causal_mask = [](int N) {
    auto indices = mx::arange(N);
    return mx::expand_dims(indices, 1) >= indices;
  };

  // Helper to expand the cache and mask
  auto expand = [](auto &args, auto &mask) {
    constexpr int cache_step_size = 256;
    int cache_size = args[1].shape(-2);
    int new_size =
        cache_step_size * ((cache_size + cache_step_size) / cache_step_size);
    for (auto it = args.begin() + 1; it != args.end(); ++it) {
      auto &x = *it;
      auto shape = x.shape();
      shape[2] = new_size;
      auto new_x = mx::zeros(shape, x.dtype());
      shape[2] = cache_size;
      *it =
          mx::slice_update(new_x, x, mx::Shape(x.ndim(), 0), std::move(shape));
    }
    mask =
        mx::slice_update(mx::full({new_size}, false), mask, {0}, {cache_size});
  };

  auto tic = time_now();
  float prompt_time;
  int n = 0;

  mx::Args args;
  {
    args = model({y, create_causal_mask(y.size())});
    auto logits = args[0];
    logits = slice(logits, {0, -1, 0}, logits.shape());
    y = argmax(logits, -1);
    async_eval(y);
  }

  auto offset = mx::array(prompt_size, mx::uint32);
  std::vector<int> tokens;

  auto mask = mx::full({prompt_size}, true);
  expand(args, mask);

  for (; n < max_tokens; ++n) {
    // Start next token decoding if needed
    if (n < max_tokens - 1) {
      args[0] = y;
      auto m = prompt_size + n;
      if (mask.size() <= m) {
        expand(args, mask);
      }
      mask = mx::slice_update(mask, mx::array(true), {m}, {m + 1});
      args.push_back(offset);
      args.push_back(mask);
      args = model(args);
      args[0] = argmax(args[0], -1);
      offset = offset + 1u;
      async_eval(args[0]);
    }

    auto token = y.item<int>();
    if (token == tokenizer.eos_token_id()) {
      break;
    }
    tokens.push_back(token);
    auto [result, complete] = tokenizer.try_decode(tokens);
    if (complete) {
      std::cout << result << std::flush;
      tokens.clear();
    }
    if (n == 0) {
      prompt_time = seconds(time_now() - tic);
      tic = time_now();
    }

    if (n < max_tokens - 1) {
      y = args[0];
    }
  }
  auto result = tokenizer.decode(tokens);
  std::cout << result << std::flush;

  auto gen_time = seconds(time_now() - tic);
  std::cout << std::endl;
  std::cout << std::setprecision(5) << "Prompt toks/sec "
            << prompt_size / prompt_time << "\nGeneration toks/sec "
            << (n + 1) / gen_time << std::endl;
}
