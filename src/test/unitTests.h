
#pragma once

#include <gtest/gtest.h>

#include <iostream>
#include <math.h>
#include <random>
#include <vector>
#include <random>

#include <cublas_v2.h>

#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>

// #include <cutlass/cutlass.h>
// #include <cutlass/numeric_types.h>
// #include <cutlass/core_io.h>
// #include <cutlass/gemm/device/gemm.h>
// #include <cutlass/gemm/device/gemm_splitk_parallel.h>
// #include <cutlass/util/host_tensor.h>
// #include <cutlass/util/tensor_view_io.h>

#include "../globals.h"
#include "../mpc/RSS.h"
#include "../util/Profiler.h"
#include "../util/util.cuh"
#include "../gpu/bitwise.cuh"
#include "../gpu/convolution.cuh"
#include "../gpu/DeviceData.h"
#include "../gpu/matrix.cuh"

extern int partyNum;
extern Profiler func_profiler;
extern Profiler comm_profiler;

extern size_t INPUT_SIZE, LAST_LAYER_SIZE, WITH_NORMALIZATION;
extern void getBatch(std::ifstream &, std::istream_iterator<double> &, std::vector<double> &);

int runTests(int argc, char **argv);

