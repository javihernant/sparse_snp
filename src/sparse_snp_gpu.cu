#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern "C" { // nvcc compiles in C++
#include <sparse_snp.h>
}

#include <omp.h>
#include <cuda.h>
#include <cub/cub.cuh>

typedef unsigned int uint;

using namespace std;
using namespace cub;

