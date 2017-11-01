#pragma once
inline __device__ uint64_t rotate180(uint64_t x) {
  return __brevll(x);
}

inline __device__ uint64_t flipVertical(uint64_t x) {
  x = (x >> 32) | (x << 32);
  x = ((x >> 16) & UINT64_C(0x0000FFFF0000FFFF)) | ((x << 16) & UINT64_C(0xFFFF0000FFFF0000));
  x = ((x >>  8) & UINT64_C(0x00FF00FF00FF00FF)) | ((x <<  8) & UINT64_C(0xFF00FF00FF00FF00));
  return x;
}

inline __device__ uint64_t mirrorHorizontal(uint64_t x) {
  x = ((x >>  4) & UINT64_C(0x0F0F0F0F0F0F0F0F)) | ((x <<  4) & UINT64_C(0xF0F0F0F0F0F0F0F0));
  x = ((x >>  2) & UINT64_C(0x3333333333333333)) | ((x <<  2) & UINT64_C(0xCCCCCCCCCCCCCCCC));
  x = ((x >>  1) & UINT64_C(0x5555555555555555)) | ((x <<  1) & UINT64_C(0xAAAAAAAAAAAAAAAA));
  return x;
}

inline __device__ uint64_t delta_swap(uint64_t bits, uint64_t mask, int delta) {
  uint64_t tmp = mask & (bits ^ (bits << delta));
  return bits ^ tmp ^ (tmp >> delta);
}

inline __device__ uint64_t flipDiagA1H8(uint64_t bits) {
  uint64_t mask1 = UINT64_C(0x5500550055005500);
  uint64_t mask2 = UINT64_C(0x3333000033330000);
  uint64_t mask3 = UINT64_C(0x0f0f0f0f00000000);
  bits = delta_swap(bits, mask3, 28);
  bits = delta_swap(bits, mask2, 14);
  return delta_swap(bits, mask1, 7);
}

inline __device__ uint64_t flipDiagA8H1(uint64_t bits) {
  uint64_t mask1 = UINT64_C(0xaa00aa00aa00aa00);
  uint64_t mask2 = UINT64_C(0xcccc0000cccc0000);
  uint64_t mask3 = UINT64_C(0xf0f0f0f000000000);
  bits = delta_swap(bits, mask3, 36);
  bits = delta_swap(bits, mask2, 18);
  return delta_swap(bits, mask1, 9);
}

inline __device__ uint64_t rotate90clockwise(uint64_t x) {
  return flipVertical(flipDiagA8H1(x));
}

inline __device__ uint64_t rotate90antiClockwise(uint64_t x) {
  return flipVertical(flipDiagA1H8(x));
}

inline __device__ uint64_t parallel_bit_extract(uint64_t x, uint64_t mask) {
  uint64_t res = 0;
  int cnt = 0;
  uint64_t bit = 0;
  for (; mask; mask &= mask+bit) {
    bit = mask & -mask;
    res |= (x & mask & ~(mask + bit)) >> (__popcll(bit-1) - cnt);
    cnt += __popcll(mask & ~(mask + bit));
  }
  return res;
}

struct board_gpu {
  uint64_t me, op;
  __host__ __device__ board_gpu() = default;
  __host__ __device__ board_gpu(uint64_t me, uint64_t op)
    : me(me), op(op) {}
};

inline __host__ __device__ board_gpu pass(const board_gpu &bd) {
  return board_gpu(bd.op, bd.me);
}

inline __device__ board_gpu flipDiagA8H1(const board_gpu &bd) {
  return board_gpu(flipDiagA8H1(bd.me), flipDiagA8H1(bd.op));
}

inline __device__ board_gpu rotate90clockwise(const board_gpu &bd) {
  return board_gpu(rotate90clockwise(bd.me), rotate90clockwise(bd.op));
}

inline __device__ board_gpu mirrorHorizontal(const board_gpu &bd) {
  return board_gpu(mirrorHorizontal(bd.me), mirrorHorizontal(bd.op));
}

inline __device__ board_gpu parallel_bit_extract(board_gpu bd, uint64_t mask) {
  return board_gpu(parallel_bit_extract(bd.me, mask), parallel_bit_extract(bd.op, mask));
}

__constant__ uint64_t mask1[4] = {
  0x0080808080808080ULL,
  0x7f00000000000000ULL,
  0x0102040810204000ULL,
  0x0040201008040201ULL
};
__constant__ uint64_t mask2[4] = {
  0x0101010101010100ULL,
  0x00000000000000feULL,
  0x0002040810204080ULL,
  0x8040201008040200ULL
};

inline __device__ uint64_t flip(const board_gpu &bd, int pos, int index) {
  uint64_t om = bd.op;
  if (index) om &= 0x7E7E7E7E7E7E7E7EULL;
  uint64_t mask = mask1[index] >> (63 - pos);
  uint64_t outflank = (0x8000000000000000ULL >> __clzll(~om & mask)) & bd.me;
  uint64_t flipped = (-outflank << 1) & mask;
  mask = mask2[index] << pos;
  outflank = mask & ((om | ~mask) + 1) & bd.me;
  flipped |= (outflank - (outflank != 0)) & mask;
  return flipped;
}

inline __device__ uint64_t flip_all(const board_gpu &bd, int pos) {
  return flip(bd, pos, 0) | flip(bd, pos, 1) | flip(bd, pos, 2) | flip(bd, pos, 3);
}

inline __device__ int mobility_count(const board_gpu &bd) {
  uint64_t bits = ~(bd.me | bd.op);
  int cnt = 0;
  for (; bits; bits &= bits-1) {
    uint64_t bit = bits & -bits;
    int pos = __popcll(bit-1);
    if (flip_all(bd, pos)) ++cnt;
  }
  return cnt;
}

inline __host__ __device__ bool operator==(const board_gpu &lhs, const board_gpu &rhs) {
  return lhs.me == rhs.me && lhs.op == rhs.op;
}
