#include "get_input.hpp"
#include <iostream>
#include <fstream>
#include "board.hpp"
#include "subboard.hpp"
#include "immintrin.h"

std::array<uint64_t, patterns> bits_h;
std::array<uint32_t, patterns+1> offset_h;

void init() {
  bit_manipulations::init();
  subboard::init();
  std::ifstream ifs("subboard.txt");
  std::string line;
  uint32_t offset = 0;
  for (int i = 0; i < patterns; ++i) {
    uint64_t bit = 0;
    for (int j = 0; j < 8; ++j) {
      std::getline(ifs, line);
      for (int k = 0; k < 8; ++k) {
        if (line[k] == 'o') {
          bit |= UINT64_C(1) << (j*8 + k);
        }
      }
    }
    bits_h[i] = bit;
    offset_h[i] = offset;
    offset += pow3(_popcnt64(bit));
    if (i != patterns-1) std::getline(ifs, line);
  }
  offset_h[patterns] = offset;
}

std::array<uint32_t, pattern_length+1> serialize_b81(const std::string& b81) {
  board bd = bit_manipulations::toBoard(b81);
  return subboard::serialize<patterns>(bd, bits_h);
}

std::pair<uint64_t, uint64_t> to_board(const std::string &b81) {
  board bd = bit_manipulations::toBoard(b81);
  return std::make_pair(bd.black(), bd.white());
}
