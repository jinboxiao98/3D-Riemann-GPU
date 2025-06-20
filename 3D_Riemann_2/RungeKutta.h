#ifndef RUNGE_KUTTA_H
#define RUNGE_KUTTA_H

__global__ void RungeKuttaKernel_1(
          double* __restrict__ const U1,
    const double* __restrict__ const U,     const double* __restrict__ const RHS
) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;

    if (
        i < BUFFER || i >= NX + BUFFER ||
        j < BUFFER || j >= NY + BUFFER ||
        k < BUFFER || k >= NZ + BUFFER
    ) return;

    const int idx = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + i;

    U1[idx] = U[idx] + DT * RHS[idx];

}

__global__ void RungeKuttaKernel_2(
          double* __restrict__ const U2,
    const double* __restrict__ const U,     const double* __restrict__ const U1,    const double* __restrict__ const RHS
) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;

    if (
        i < BUFFER || i >= NX + BUFFER ||
        j < BUFFER || j >= NY + BUFFER ||
        k < BUFFER || k >= NZ + BUFFER
    ) return;

    const int idx = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + i;

    U2[idx] = 0.75 * U[idx] + 0.25 * U1[idx] + 0.25 * DT * RHS[idx];

}

__global__ void RungeKuttaKernel_3(
          double* __restrict__ const U,
    const double* __restrict__ const U2,    const double* __restrict__ const RHS
) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;

    if (
        i < BUFFER || i >= NX + BUFFER ||
        j < BUFFER || j >= NY + BUFFER ||
        k < BUFFER || k >= NZ + BUFFER
    ) return;

    const int idx = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + i;

    U[idx] = (1.0/3.0) * U[idx] + (2.0/3.0) * U2[idx] + (2.0/3.0) * DT * RHS[idx];
    
}

#endif