#include <array>
#include <fstream>
#include <iostream>
#include <cuda_profiler_api.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/extrema.h>

constexpr int pattern_num = 46;
constexpr int threadPerBlock = 256;
constexpr float eps = 1.0;

constexpr int pow3(int index) {
  return index == 0 ? 1 : 3 * pow3(index-1);
}

enum Pattern {
  HORIZONTAL1,
  HORIZONTAL2,
  HORIZONTAL3,
  DIAG8,
  DIAG7,
  DIAG6,
  DIAG5,
  DIAG4,
  CORNER2X5,
  CORNER3X3,
  EDGE2X
};

constexpr std::array<int, 11> size_ary    = {8, 8, 8, 8, 7, 6, 5, 4, 10, 9, 10};
constexpr std::array<int, 11> variant_ary = {4, 4, 4, 2, 4, 4, 4, 4,  8, 4,  4};
constexpr int index_begin(const int part) {
  return part == 0 ? 0 : (index_begin(part-1) + pow3(size_ary[part-1]));
}
constexpr int variant_index_begin(const int part) {
  return part == 0 ? 0 : (variant_index_begin(part-1) + variant_ary[part-1]);
}

constexpr int m = index_begin(size_ary.size());
constexpr int M = m + 3;

texture<int> texIndices;
texture<int> texPuttable;
texture<int> texPuttableOp;

__global__ void spmat_vec_impl(const size_t pitch,
    const float * const x, float * const b, const int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  while (index < n) {
    float res = 0;
    for (int i = 0; i < pattern_num; ++i) {
      res += x[tex1Dfetch(texIndices, i*pitch + index)];
    }
    res += tex1Dfetch(texPuttable, index) * x[m];
    res += tex1Dfetch(texPuttableOp, index) * x[m+1];
    res += x[m+2];
    b[index] = res;
    index += blockDim.x * gridDim.x;
  }
}

__global__ void trans_spmat_vec_impl(const size_t pitch,
    float * const x, const float * const b, const int n) {
  __shared__ float cache1[threadPerBlock], cache2[threadPerBlock], cache3[threadPerBlock];
  __shared__ float diag4_cache[pow3(4)*32];
  __shared__ float diag5_cache[pow3(5)];
  for (int k = threadIdx.x; k < pow3(4)*32; k += blockDim.x) diag4_cache[k] = 0;
  for (int k = threadIdx.x; k < pow3(5); k += blockDim.x) diag5_cache[k] = 0;
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.x;
  int jm32 = j % 32;
  cache1[j] = cache2[j] = cache3[j] = 0;
  while (index < n) {
    const float b_value = b[index];
    for (int i = 0; i < pattern_num; ++i) {
      if (i < variant_index_begin(Pattern::DIAG5)) {
        atomicAdd(x + tex1Dfetch(texIndices, i*pitch + index), b_value);
      } else if (i < variant_index_begin(Pattern::DIAG4)) {
        atomicAdd(diag5_cache + tex1Dfetch(texIndices, i*pitch + index) - index_begin(Pattern::DIAG5), b_value);
      } else if (i < variant_index_begin(Pattern::CORNER2X5)) {
        atomicAdd(diag4_cache + (tex1Dfetch(texIndices, i*pitch + index) - index_begin(Pattern::DIAG4)) * 32 + jm32, b_value);
      } else {
        atomicAdd(x + tex1Dfetch(texIndices, i*pitch + index), b_value);
      }
    }
    cache1[j] += tex1Dfetch(texPuttable, index) * b_value;
    cache2[j] += tex1Dfetch(texPuttableOp, index) * b_value;
    cache3[j] += b_value;
    index += blockDim.x * gridDim.x;
  }
  __syncthreads();
  for (int k = threadIdx.x; k < pow3(4); k += blockDim.x) {
    for (int l = 1; l < 32; ++l)
      diag4_cache[k*32] += diag4_cache[k*32 + l];
    atomicAdd(x + index_begin(Pattern::DIAG4) + k, diag4_cache[k*32]);
  }
  for (int k = threadIdx.x; k < pow3(5); k += blockDim.x)
    atomicAdd(x + index_begin(Pattern::DIAG5) + k, diag5_cache[k]);
  int i = blockDim.x / 2;
  while (i) {
    if (j < i) {
      cache1[j] += cache1[j + i];
      cache2[j] += cache2[j + i];
      cache3[j] += cache3[j + i];
    }
    __syncthreads();
    i /= 2;
  }
  if (j == 0) {
    atomicAdd(x + m, cache1[0]);
    atomicAdd(x + m + 1, cache2[0]);
    atomicAdd(x + m + 2, cache3[0]);
  }
}

__global__ void sqnorm_impl(const float * const vec, const int n, float * const resary) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ float cache[threadPerBlock];
  float sum = 0;
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

__global__ void fma1_impl(float * const vec1, const float * const vec2, const float scalar, const int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  while (index < n) {
    vec1[index] += scalar * vec2[index];
    index += blockDim.x * gridDim.x;
  }
}

__global__ void fma2_impl(const float * const vec1, float * const vec2, const float scalar, const int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  while (index < n) {
    vec2[index] = scalar * vec2[index] + vec1[index];
    index += blockDim.x * gridDim.x;
  }
}

__global__ void sub_impl(float * const vec1, const float * const vec2, const int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  while (index < n) {
    vec1[index] -= vec2[index];
    index += blockDim.x * gridDim.x;
  }
}

float *sqnorm_ary;
float *sqnorm_ary_dev;

void sqnorm_init() {
  sqnorm_ary = (float*)malloc(sizeof(float) * 1024);
  cudaMalloc(reinterpret_cast<void**>(&sqnorm_ary_dev), sizeof(float) * 1024);
}

void sqnorm_exit() {
  cudaFree(sqnorm_ary_dev);
  free(sqnorm_ary);
}

__host__ float sqnorm(const thrust::device_vector<float> &vec) {
  sqnorm_impl<<<1024, threadPerBlock>>>(vec.data().get(), vec.size(), sqnorm_ary_dev);
  cudaMemcpy(sqnorm_ary, sqnorm_ary_dev, sizeof(float) * 1024, cudaMemcpyDeviceToHost);
  float res = 0;
  for (int i = 0; i < 1024; ++i) res += sqnorm_ary[i];
  return res;
}

// vec1 += scalar * vec2
__host__ void fma1(thrust::device_vector<float> &vec1,
    const thrust::device_vector<float> &vec2, float scalar) {
  fma1_impl<<<1024, threadPerBlock>>>(vec1.data().get(), vec2.data().get(), scalar, vec1.size());
}

// vec2 = scalar * vec2 + vec1
__host__ void fma2(const thrust::device_vector<float> &vec1,
    thrust::device_vector<float> &vec2, float scalar) {
  fma2_impl<<<1024, threadPerBlock>>>(vec1.data().get(), vec2.data().get(), scalar, vec1.size());
}

// vec1 -= vec2
__host__ void sub(thrust::device_vector<float> &vec1,
    const thrust::device_vector<float> &vec2) {
  sub_impl<<<1024, threadPerBlock>>>(vec1.data().get(), vec2.data().get(), vec1.size());
}

__host__ float abs(const thrust::device_vector<float> &vec) {
  return sqrt(sqnorm(vec));
}

struct SpMat {
  int n;
  int *indices;
  size_t pitch;
  int *puttable;
  int *puttable_op;
};

void trans_spmat_vec(const SpMat &mat,
    thrust::device_vector<float> &x,
    const thrust::device_vector<float> &b) {
  x.assign(x.size(), 0.0);
  trans_spmat_vec_impl<<<1024, threadPerBlock>>>(mat.pitch, x.data().get(), b.data().get(), mat.n);
}

void spmat_vec(const SpMat &mat,
    const thrust::device_vector<float> &x,
    thrust::device_vector<float> &b) {
  spmat_vec_impl<<<1024, threadPerBlock>>>(mat.pitch, x.data().get(), b.data().get(), mat.n);
}

__host__ thrust::host_vector<float> CGLSmethod(const SpMat &mat, const thrust::host_vector<float> &b, const int loop_count) {
  cudaEvent_t start, stop, start2, stop2;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);
  thrust::device_vector<float> b_dev = b;
  thrust::device_vector<float> r(M, 0.0);
  cudaEventRecord(start, 0);
  trans_spmat_vec(mat, r, b_dev);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);
  std::cerr << elapsed_time << std::endl;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  thrust::device_vector<float> x(M, 0.0);
  thrust::device_vector<float> d = b_dev;
  thrust::device_vector<float> p = r;
  thrust::device_vector<float> t(mat.n, 0.0);
  cudaEventRecord(start2, 0);
  spmat_vec(mat, p, t);
  cudaEventRecord(stop2, 0);
  cudaEventSynchronize(stop2);
  cudaEventElapsedTime(&elapsed_time, start2, stop2);
  std::cerr << elapsed_time << std::endl;
  cudaEventDestroy(start2);
  cudaEventDestroy(stop2);
  float sqn_r = sqnorm(r);
  std::cerr << sqn_r << std::endl;
  thrust::device_vector<float> tmp(mat.n);
  for (int i = 0; i < loop_count; ++i) {
    float alpha = sqn_r / sqnorm(t);
    fma1(x, p, alpha);
    fma1(d, t, -alpha);
    trans_spmat_vec(mat, r, d);
    float diff = abs(r);
    if (diff < eps) break;
    if ((i % 10) == 0) {
      spmat_vec(mat, x, tmp);
      sub(tmp, b);
      std::cerr << (abs(tmp)/sqrt(tmp.size())) << std::endl;
    }
    float old_sqn_r = sqn_r;
    sqn_r = sqnorm(r);
    float beta = sqn_r / old_sqn_r;
    fma2(r, p, beta);
    spmat_vec(mat, p, t);
  }
  return thrust::host_vector<float>(x);
}

int main(int argc, char **argv) {
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
  std::cerr << m << std::endl;
  std::cerr << variant_index_begin(11) << std::endl;
  int n;
  ifs >> n;
  int *indices_dev = nullptr;
  size_t pitch;
  cudaMallocPitch(reinterpret_cast<void**>(&indices_dev), &pitch, sizeof(int) * n, pattern_num);
  std::cerr << pitch << std::endl;
  int *indices_host = (int*)malloc(pitch * pattern_num);
  thrust::host_vector<int> puttable(n);
  thrust::host_vector<int> puttable_op(n);
  thrust::host_vector<float> b(n);
  thrust::host_vector<thrust::host_vector<int>> indices_trans(m);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < pattern_num; ++j) {
      int index;
      ifs >> index;
      indices_host[j*pitch/sizeof(int) + i] = index;
      indices_trans[index].push_back(i);
    }
    ifs >> puttable[i] >> puttable_op[i] >> b[i];
  }
  thrust::device_vector<int> puttable_dev = puttable;
  thrust::device_vector<int> puttable_op_dev = puttable_op;
  cudaMemcpy(indices_dev, indices_host, pitch * pattern_num, cudaMemcpyHostToDevice);
  sqnorm_init();
  SpMat mat = {n, indices_dev, pitch/sizeof(int), puttable_dev.data().get(), puttable_op_dev.data().get()};
  cudaBindTexture(nullptr, texIndices, indices_dev, pitch * pattern_num);
  cudaBindTexture(nullptr, texPuttable, puttable_dev.data().get(), sizeof(int) * n);
  cudaBindTexture(nullptr, texPuttableOp, puttable_op_dev.data().get(), sizeof(int) * n);
  std::cerr << "start" << std::endl;
  thrust::host_vector<float> res = CGLSmethod(mat, b, loop_count);
  std::cerr << "end" << std::endl;
  cudaFree(indices_dev);
  free(indices_host);
  for (const auto &v : res) {
    ofs << v << '\n';
  }
  ofs << std::flush;
  sqnorm_exit();
  cudaUnbindTexture(texIndices);
  cudaUnbindTexture(texPuttable);
  cudaUnbindTexture(texPuttableOp);
  return 0;
}
