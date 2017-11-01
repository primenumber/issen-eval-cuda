#pragma once
#include <array>
#include <utility>

void init();

constexpr int patterns = 13;
extern std::array<uint64_t, patterns> bits_h;
extern std::array<uint32_t, patterns+1> offset_h;
constexpr int pattern_length = 8*patterns;
std::array<uint32_t, pattern_length+1> serialize_b81(const std::string &);
std::pair<uint64_t, uint64_t> to_board(const std::string &);

constexpr int pow3(int index) {
  return index == 0 ? 1 : 3 * pow3(index-1);
}
