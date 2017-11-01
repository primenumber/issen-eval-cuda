#include <array>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <cuda_profiler_api.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/extrema.h>
#include "get_input.hpp"
#include "board_gpu.cuh"

constexpr int threadPerBlock = 256;
constexpr double eps = 1.0;

constexpr int patterns_shared = 2;
constexpr int buffer_size = pow3(6) + pow3(5);
constexpr int patterns_global = patterns - patterns_shared;

constexpr int base3_bits = 12;

__device__ uint32_t toBase3(uint32_t x, uint32_t y, const uint32_t * const base3) {
  return __ldg(base3 + x) + 2 * __ldg(base3 + y);
}

__constant__ uint64_t bits[patterns];
__constant__ uint32_t offset[patterns+1];

__global__ void spmat_vec_impl(const board_gpu * const boards, const uint32_t * const base3,
    const double * const x, double * const b, const int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  while (index < n) {
    board_gpu bd = boards[index];
    board_gpu transed_bd[8];
    for (int i = 0; i < 4; ++i) {
      transed_bd[i] = bd;
      transed_bd[i+4] = mirrorHorizontal(bd);
      bd = rotate90clockwise(bd);
    }
    double res = 0;
    for (int i = 0; i < patterns; ++i) {
      for (int j = 0; j < 8; ++j) {
        board_gpu pbe = parallel_bit_extract(transed_bd[j], bits[i]);
        res += __ldg(x + toBase3(pbe.me, pbe.op, base3) + offset[i]);
      }
    }
    res += mobility_count(bd) * x[offset[patterns]];
    res += mobility_count(pass(bd)) * x[offset[patterns]+1];
    res += x[offset[patterns]+2];
    b[index] = res;
    index += blockDim.x * gridDim.x;
  }
}

__global__ void trans_spmat_vec_impl(const board_gpu * const boards, const uint32_t * const base3,
    double * const x, const double * const b, const int n) {
  __shared__ double x_s[buffer_size];
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int id = threadIdx.x;
  for (int k = id; k < buffer_size; k += blockDim.x) x_s[k] = 0;
  __syncthreads();
  double me_sum = 0;
  double op_sum = 0;
  double const_sum = 0;
  while (index < n) {
    board_gpu bd = boards[index];
    board_gpu transed_bd[8];
    for (int i = 0; i < 4; ++i) {
      transed_bd[i] = bd;
      transed_bd[i+4] = mirrorHorizontal(bd);
      bd = rotate90clockwise(bd);
    }
    const double b_value = b[index];
    for (int i = 0; i < patterns; ++i) {
      for (int j = 0; j < 8; ++j) {
        board_gpu pbe = parallel_bit_extract(transed_bd[j], bits[i]);
        if (i < patterns_global) {
          atomicAdd(x + toBase3(pbe.me, pbe.op, base3) + offset[i], b_value);
        } else {
          atomicAdd(x_s + toBase3(pbe.me, pbe.op, base3) + offset[i] - offset[patterns_global], b_value);
        }
      }
    }
    me_sum += mobility_count(bd) * b_value;
    op_sum += mobility_count(pass(bd)) * b_value;
    const_sum += b_value;
    index += blockDim.x * gridDim.x;
  }
  __syncthreads();
  for (int k = id; k < buffer_size; k += blockDim.x) {
    atomicAdd(x + k + offset[patterns_global], x_s[k]);
  }
  int i = blockDim.x / 2;
  __shared__ double psum[threadPerBlock][3];
  psum[id][0] = me_sum;
  psum[id][1] = op_sum;
  psum[id][2] = const_sum;
  __syncthreads();
  while (i) {
    if (id < i) {
      psum[id][0] += psum[id + i][0];
      psum[id][1] += psum[id + i][1];
      psum[id][2] += psum[id + i][2];
    }
    __syncthreads();
    i /= 2;
  }
  if (id == 0) {
    atomicAdd(x + offset[patterns], psum[0][0]);
    atomicAdd(x + offset[patterns] + 1, psum[0][1]);
    atomicAdd(x + offset[patterns] + 2, psum[0][2]);
  }
}

__global__ void sqnorm_impl(const double * const vec, const int n, double * const resary) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ double cache[threadPerBlock];
  double sum = 0;
  while (index < n) {
    sum += vec[index] * vec[index];
    index += blockDim.x * gridDim.x;
  }
  int j = threadIdx.x;
  cache[j] = sum;
  __syncthreads();
  int i = blockDim.x / 2;
  while (i) {
    if (j < i) cache[j] += cache[j+i];
    __syncthreads();
    i /= 2;
  }
  if (j == 0) resary[blockIdx.x] = cache[0];
}

__global__ void l1norm_impl(const double * const vec, const int n, double * const resary) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ double cache[threadPerBlock];
  double sum = 0;
  while (index < n) {
    sum += abs(vec[index]);
    index += blockDim.x * gridDim.x;
  }
  int j = threadIdx.x;
  cache[j] = sum;
  __syncthreads();
  int i = blockDim.x / 2;
  while (i) {
    if (j < i) cache[j] += cache[j+i];
    __syncthreads();
    i /= 2;
  }
  if (j == 0) resary[blockIdx.x] = cache[0];
}

__global__ void max_impl(const double * const vec, const int n, double * const resary) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ double cache[threadPerBlock];
  double res = 0;
  while (index < n) {
    res = max(res, abs(vec[index]));
    index += blockDim.x * gridDim.x;
  }
  int j = threadIdx.x;
  cache[j] = res;
  __syncthreads();
  int i = blockDim.x / 2;
  while (i) {
    if (j < i) cache[j] = max(cache[j], cache[j+i]);
    __syncthreads();
    i /= 2;
  }
  if (j == 0) resary[blockIdx.x] = cache[0];
}

__global__ void fma1_impl(double * const vec1, const double * const vec2, const double scalar, const int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  while (index < n) {
    vec1[index] += scalar * vec2[index];
    index += blockDim.x * gridDim.x;
  }
}

__global__ void fma2_impl(const double * const vec1, double * const vec2, const double scalar, const int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  while (index < n) {
    vec2[index] = scalar * vec2[index] + vec1[index];
    index += blockDim.x * gridDim.x;
  }
}

__global__ void sub_impl(double * const vec1, const double * const vec2, const int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  while (index < n) {
    vec1[index] -= vec2[index];
    index += blockDim.x * gridDim.x;
  }
}

double *norm_ary;
double *norm_ary_dev;
uint32_t *base3_d;

void init_mems() {
  // norm
  norm_ary = (double*)malloc(sizeof(double) * 1024);
  cudaMalloc(reinterpret_cast<void**>(&norm_ary_dev), sizeof(double) * 1024);
  // base3
  uint32_t base3_h[1 << base3_bits];
  for (int i = 0; i < (1 << base3_bits); ++i) {
    base3_h[i] = 0;
    for (int j = 0; j < base3_bits; ++j) {
      if ((i >> j) & 1) {
        base3_h[i] += pow3(j);
      }
    }
  }
  cudaMalloc(&base3_d, sizeof(uint32_t) * (1 << base3_bits));
  cudaMemcpy(base3_d, base3_h, sizeof(uint32_t) * (1 << base3_bits), cudaMemcpyHostToDevice);
  // offset/bits
  cudaMemcpyToSymbol(bits, bits_h.data(), patterns * sizeof(uint64_t));
  cudaMemcpyToSymbol(offset, offset_h.data(), (patterns+1) * sizeof(uint32_t));
}

void exit_mems() {
  cudaFree(norm_ary_dev);
  free(norm_ary);
  cudaFree(base3_d);
}

__host__ double sqnorm(const thrust::device_vector<double> &vec) {
  sqnorm_impl<<<1024, threadPerBlock>>>(vec.data().get(), vec.size(), norm_ary_dev);
  cudaMemcpy(norm_ary, norm_ary_dev, sizeof(double) * 1024, cudaMemcpyDeviceToHost);
  double res = 0;
  for (int i = 0; i < 1024; ++i) res += norm_ary[i];
  return res;
}

__host__ double l1norm(const thrust::device_vector<double> &vec) {
  l1norm_impl<<<1024, threadPerBlock>>>(vec.data().get(), vec.size(), norm_ary_dev);
  cudaMemcpy(norm_ary, norm_ary_dev, sizeof(double) * 1024, cudaMemcpyDeviceToHost);
  double res = 0;
  for (int i = 0; i < 1024; ++i) res += norm_ary[i];
  return res;
}

__host__ double max(const thrust::device_vector<double> &vec) {
  max_impl<<<1024, threadPerBlock>>>(vec.data().get(), vec.size(), norm_ary_dev);
  cudaMemcpy(norm_ary, norm_ary_dev, sizeof(double) * 1024, cudaMemcpyDeviceToHost);
  double res = 0;
  for (int i = 0; i < 1024; ++i) {
    res = std::max(res, norm_ary[i]);
  }
  return res;
}

// vec1 += scalar * vec2
__host__ void fma1(thrust::device_vector<double> &vec1,
    const thrust::device_vector<double> &vec2, double scalar) {
  fma1_impl<<<1024, threadPerBlock>>>(vec1.data().get(), vec2.data().get(), scalar, vec1.size());
}

// vec2 = scalar * vec2 + vec1
__host__ void fma2(const thrust::device_vector<double> &vec1,
    thrust::device_vector<double> &vec2, double scalar) {
  fma2_impl<<<1024, threadPerBlock>>>(vec1.data().get(), vec2.data().get(), scalar, vec1.size());
}

// vec1 -= vec2
__host__ void sub(thrust::device_vector<double> &vec1,
    const thrust::device_vector<double> &vec2) {
  sub_impl<<<1024, threadPerBlock>>>(vec1.data().get(), vec2.data().get(), vec1.size());
}

__host__ double abs(const thrust::device_vector<double> &vec) {
  return sqrt(sqnorm(vec));
}

struct SpMat {
  int n;
  board_gpu *boards;
  uint32_t *base3;
};

void trans_spmat_vec(const SpMat &mat,
    thrust::device_vector<double> &x,
    const thrust::device_vector<double> &b) {
  x.assign(x.size(), 0.0);
  trans_spmat_vec_impl<<<4096, threadPerBlock>>>(mat.boards, mat.base3, x.data().get(), b.data().get(), mat.n);
}

void spmat_vec(const SpMat &mat,
    const thrust::device_vector<double> &x,
    thrust::device_vector<double> &b) {
  spmat_vec_impl<<<1024, threadPerBlock>>>(mat.boards, mat.base3, x.data().get(), b.data().get(), mat.n);
}

__host__ thrust::host_vector<double> CGLSmethod(const SpMat &mat, const thrust::host_vector<double> &b, const int loop_count) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  thrust::device_vector<double> b_dev = b;
  thrust::device_vector<double> r(offset_h[patterns]+3, 0.0);
  cudaEventRecord(start, 0);
  trans_spmat_vec(mat, r, b_dev);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);
  std::cerr << "trans_spmat_vec: " << elapsed_time << "ms" << std::endl;
  thrust::device_vector<double> x(offset_h[patterns]+3, 0.0);
  thrust::device_vector<double> d = b_dev;
  thrust::device_vector<double> p = r;
  thrust::device_vector<double> t(mat.n, 0.0);
  cudaEventRecord(start, 0);
  spmat_vec(mat, p, t);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  std::cerr << "spmat_vec: " << elapsed_time << "ms" << std::endl;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  double sqn_r = sqnorm(r);
  std::cerr << "initial error: " << (sqn_r/sqrt(r.size())) << std::endl;
  thrust::device_vector<double> tmp(mat.n);
  for (int i = 0; i < loop_count; ++i) {
    double alpha = sqn_r / sqnorm(t);
    fma1(x, p, alpha);
    fma1(d, t, -alpha);
    trans_spmat_vec(mat, r, d);
    double diff = abs(r);
    if (diff < eps) break;
    double old_sqn_r = sqn_r;
    sqn_r = sqnorm(r);
    double beta = sqn_r / old_sqn_r;
    fma2(r, p, beta);
    spmat_vec(mat, p, t);
    if ((i % 10) == 0) {
      spmat_vec(mat, x, tmp);
      sub(tmp, b);
      std::cerr << i << " : error=" << (abs(tmp)/sqrt(tmp.size())) << ", avgdiff=" << (l1norm(tmp)/tmp.size()) << ", maxdelta=" << abs(alpha) * max(p) << std::endl;
    }
  }
  return thrust::host_vector<double>(x);
}

std::string to_str(const board_gpu &bd) {
  std::string res;
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      int index = i*8 + j;
      if ((bd.me >> index) & 1) {
        res += 'x';
      } else if ((bd.op >> index) & 1) {
        res += 'o';
      } else {
        res += ' ';
      }
    }
    res += '\n';
  }
  return res;
}

int main(int argc, char **argv) {
  std::ios::sync_with_stdio(false);
  if (argc != 4) {
    std::cerr << argv[0] << " LOOP_COUNT INPUT OUTPUT" << std::endl;
    exit(EXIT_FAILURE);
  }
  const int loop_count = std::atoi(argv[1]);
  std::ifstream ifs(argv[2]);
  if (ifs.fail()) {
    std::cerr << "Couldn't open file: " << argv[2] << std::endl;
    exit(EXIT_FAILURE);
  }
  std::ofstream ofs(argv[3]);
  if (ofs.fail()) {
    std::cerr << "Couldn't open file: " << argv[3] << std::endl;
    exit(EXIT_FAILURE);
  }
  init();
  int n;
  ifs >> n;
  std::cerr << n << std::endl;
  board_gpu *boards = nullptr;
  cudaMallocManaged(reinterpret_cast<void**>(&boards), n * sizeof(board_gpu));
  thrust::host_vector<double> b(n);
  std::string b81;
  for (int i = 0; i < n; ++i) {
    ifs >> b81;
    auto p = to_board(b81);
    boards[i] = board_gpu(p.first, p.second);
    ifs >> b[i];
  }
  cudaMemAdvise(boards, n * sizeof(board_gpu), cudaMemAdviseSetReadMostly, 0);
  init_mems();
  SpMat mat = {n, boards, base3_d};
  std::cerr << "start" << std::endl;
  std::cerr << std::fixed << std::setprecision(6);
  thrust::host_vector<double> res = CGLSmethod(mat, b, loop_count);
  std::cerr << "end" << std::endl;
  cudaFree(boards);
  ofs << std::fixed << std::setprecision(6);
  for (const auto &v : res) {
    ofs << v << '\n';
  }
  ofs << std::flush;
  exit_mems();
  return 0;
}
