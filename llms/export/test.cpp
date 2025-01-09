// Copyright © 2024 Apple Inc.

#include "tokenizer.h"
#include <iostream>

template <typename T, typename U = T> void check(const T &x, const U &y) {
  if (x != y) {
    std::cerr << "Mismatch" << std::endl;
  }
}

void test_tokenizer(const std::string &path) {
  BPETokenizer tokenizer(path);
  check(tokenizer.encode("hello world!"), {128000, 15339, 1917, 0});
  check(tokenizer.decode({15339}), "hello");
  check(tokenizer.decode({0}), "!");
  check(tokenizer.decode({1917}), " world");
  check(tokenizer.encode("we'd  see   you say 世界你好真实好的很啊"),
        {128000, 906, 4265, 220, 1518, 256, 499, 2019, 127365, 57668, 53901,
         89151, 41073, 110085, 101600, 102856});
}

int main(int argc, char *argv[]) { test_tokenizer("."); }
