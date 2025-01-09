
#include <codecvt>
#include <filesystem>
#include <fstream>
#include <json.hpp>
#include <locale>

#include "third_party/unicode.h"
#include "tokenizer.h"

using json = nlohmann::json;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
std::pair<std::wstring, int> utf8_to_utf16(const std::string &s) {
  static std::string replace_str = std::string(1, 0xFF);
  static std::wstring replace_wstr = std::wstring(1, 0xFFFD);
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> cvt(replace_str,
                                                             replace_wstr);
  auto out = cvt.from_bytes(s);
  return {out, cvt.converted()};
}
#pragma GCC diagnostic pop

auto make_byte_decoder() {
  std::unordered_map<uint16_t, char> byte_decoder;
  std::vector<uint16_t> limits = {0,        '!',  '~' + 1, L'¡',
                                  L'¬' + 1, L'®', L'ÿ' + 1};
  char n = 0;
  for (int i = 0; i < limits.size() - 1; ++i) {
    auto start = limits[i];
    auto stop = limits[i + 1];
    if (i % 2 == 0) {
      for (int b = start; b < stop; ++b) {
        byte_decoder[256 + n++] = b;
      }
    } else {
      for (int b = start; b < stop; ++b) {
        byte_decoder[b] = b;
      }
    }
  }
  return byte_decoder;
}

auto BPETokenizer::byte_decoder_ = make_byte_decoder();

BPETokenizer::BPETokenizer(const std::string &path_) {
  auto path = std::filesystem::path(path_);
  std::ifstream ifs(path / "tokenizer.json");
  auto tokenizer = json::parse(ifs);
  auto model = tokenizer["model"];
  token_to_id_ = model["vocab"];
  id_to_token_.resize(token_to_id_.size());
  for (auto &[s, id] : token_to_id_) {
    if (id >= id_to_token_.size()) {
      id_to_token_.resize(id + 1);
    }
    id_to_token_[id] = s;
  }
  std::string type = model["type"];
  auto merges = model["merges"];
  for (auto &s : merges) {
    if (s.is_string()) {
      merges_.emplace(s, merges_.size());
    } else {
      std::string s1 = s[0];
      std::string s2 = s[1];
      merges_.emplace(s1 + " " + s2, merges_.size());
    }
  }

  auto added_tokens = tokenizer["added_tokens"];
  for (auto &added_token : added_tokens) {
    int id = added_token["id"];
    if (id >= id_to_token_.size()) {
      id_to_token_.resize(id + 1);
    }
    id_to_token_[id] = added_token["content"];
    if (id_to_token_[id] == "<|begin_of_text|>") {
      bos_id_ = id;
    } else if (id_to_token_[id] == "<|eot_id|>") {
      eos_id_ = id;
    }
  }

  // Currently hardcoded to Llama3 BPE regex
  pre_tokenizer_regex_ = {
      "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}"
      "\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
      "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"};
}

std::vector<int> BPETokenizer::encode(std::string text) const {

  auto segments = unicode_regex_split(text, pre_tokenizer_regex_);

  auto one_step_merge = [this](std::string segment, std::vector<int> &splits) {
    int merge_idx;
    int rank = INT32_MAX;
    for (int i = 0; i < splits.size() - 2; ++i) {
      auto start = splits[i];
      auto mid = splits[i + 1];
      auto end = splits[i + 2];
      std::string candidate = segment.substr(start, mid - start);
      candidate += " ";
      candidate += segment.substr(mid, end - mid);
      if (auto it = merges_.find(candidate); it != merges_.end()) {
        if (it->second < rank) {
          merge_idx = i;
          rank = it->second;
        }
      }
    }
    if (rank == INT32_MAX) {
      return false;
    }
    auto start = splits[merge_idx];
    auto mid = splits[merge_idx + 1];
    auto end = splits[merge_idx + 2];
    std::string merge_l = segment.substr(start, mid - start);
    std::string merge_r = segment.substr(mid, end - mid);
    for (int i = splits.size() - 2; i >= 0; --i) {
      auto start = splits[i];
      auto mid = splits[i + 1];
      auto end = splits[i + 2];
      if (segment.substr(start, mid - start) == merge_l &&
          segment.substr(mid, end - mid) == merge_r) {
        splits.erase(splits.begin() + i + 1);
        i -= 1;
      }
    }
    return true;
  };

  std::vector<int> ids;
  ids.push_back(bos_id_);

  // Initialize merges to integer list
  auto merge_segment = [&ids, &one_step_merge,
                        this](const std::string &segment) {
    std::vector<int> splits;
    for (int i = 0; i < segment.size(); ++i) {
      splits.push_back(i);
      if (static_cast<unsigned char>(segment[i]) >= 128) {
        i++;
      }
    }
    splits.push_back(segment.size());

    while (one_step_merge(segment, splits)) {
    };
    for (int i = 0; i < splits.size() - 1; ++i) {
      auto start = splits[i];
      auto end = splits[i + 1];
      std::string s = segment.substr(start, end - start);
      if (auto it = token_to_id_.find(s); it != token_to_id_.end()) {
        ids.push_back(it->second);
      } else {
        throw std::runtime_error("UNK ENCOUNTERED");
      }
    }
  };

  for (auto &segment : segments) {
    merge_segment(segment);
  }
  return ids;
}

std::string BPETokenizer::id_to_bytes(int id) const {
  std::string token;
  auto [wide_token, _] = utf8_to_utf16(id_to_token_[id]);
  token.resize(wide_token.size());
  for (int i = 0; i < wide_token.size(); ++i) {
    token[i] = byte_decoder_[wide_token[i]];
  }
  return token;
}

std::pair<std::string, bool>
BPETokenizer::try_decode(const std::vector<int> &ids) const {
  std::string text;
  for (auto id : ids) {
    text += id_to_bytes(id);
  }
  auto [_, converted] = utf8_to_utf16(text);
  bool complete = converted == text.size();
  text.resize(converted);
  return {text, complete};
}

std::string BPETokenizer::decode(const std::vector<int> &ids) const {
  return try_decode(ids).first;
}

int BPETokenizer::eos_token_id() const { return eos_id_; }
