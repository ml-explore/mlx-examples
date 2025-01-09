// Copyright © 2024 Apple Inc.

#include <iostream>

#include "mlxlm.h"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Must provide the model path and prompt." << std::endl;
    return 1;
  }
  auto path = std::string(argv[1]);
  auto prompt = std::string(argv[2]);

  auto model = load_model(path + "/model.mlxfn");
  auto tokenizer = load_tokenizer(path);
  generate(model, tokenizer, prompt);
}
