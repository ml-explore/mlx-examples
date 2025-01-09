// Copyright © 2024 Apple Inc.

#include <string>
#include <unordered_map>
#include <vector>

#pragma once

/** BPE Tokenizer API */
class BPETokenizer {
 public:
  BPETokenizer(const std::string& path);

  /** Encode a string of text to token integer ids. */
  std::vector<int> encode(std::string text) const;

  /** Try to decode the vector of ids to text. The text is truncated to
   * include only the fully decodable tokens. */
  std::string decode(const std::vector<int>& ids) const;

  /** Try to decode the vector of ids to text. The second return value
   * indicates if the decoding completed. The text is truncated to include
   * only the fully decodable tokens. */
  std::pair<std::string, bool> try_decode(const std::vector<int>& ids) const;

  int eos_token_id() const;

 private:
  std::unordered_map<std::string, int> token_to_id_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> merges_;
  int bos_id_;
  int eos_id_;
  static std::unordered_map<uint16_t, char> byte_decoder_;
  std::string id_to_bytes(int id) const;
  std::vector<std::string> pre_tokenizer_regex_;
};
