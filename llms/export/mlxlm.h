// Copyright © 2024 Apple Inc.

#include <mlx/mlx.h>

#include "tokenizer.h"

namespace mx = mlx::core;

std::function<mx::Args(mx::Args)> load_model(const std::string& path);

BPETokenizer load_tokenizer(const std::string& path);

struct GenerationResponse {
};

void generate(
    const std::function<mx::Args(mx::Args)>& model,
    const BPETokenizer& tokenizer,
    const std::string& prompt,
    int max_tokens = 256);
