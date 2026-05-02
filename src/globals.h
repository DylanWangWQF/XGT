
#pragma once

#include <vector>
#include <string>
#include <assert.h>
#include <limits.h>
#include <array>
#include <json.hpp>
#include <float.h>

// AES globals
#define RANDOM_COMPUTE 256	//Size of buffer for random elements
#define STRING_BUFFER_SIZE 256

// GPU configuration
#define MAX_THREADS_PER_BLOCK 32

// MPC globals
#ifndef FLOAT_PRECISION
#define FLOAT_PRECISION 15
#endif

#define PRELOAD_PATH "files/preload/"
#define TEST_PATH "files/test/"

#define MAX_JSON_DESERIALIZATION_BUFFER 1048576


extern long TREE_DEPTH_PARAM;
extern long INSTANCE_COUNT_PER_ITER;
extern long CLASS_COUNT;
extern long ATTRIBUTE_COUNT_TOTAL;
extern long TREE_DEPTH;
extern long NODE_COUNT;
extern long LEAF_COUNT;
extern long LEAF_COUNTER_SIZE;
extern long LEAF_COUNTERS_SIZE;
extern long INSTANCE_COUNT_TOTAL;
extern long INF_COUNT_TOTAL;
extern int scaleFactor;
