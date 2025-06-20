#ifndef RIEMANN_H
#define RIEMANN_H

__device__ static inline
double MINMOD(
    const double u_up2, const double u_up, const double u_down
) {
    double eps = 1e-10;
    double r = (u_down-u_up)/(u_up-u_up2 + eps);

    return u_up + 0.5*fmax(0.0, fmin(1.0, r)) * (u_up - u_up2);
}

__device__ static inline
double UPWIND(
    const double u_up2, const double u_up, const double u_down
) {
    return u_up;
}

// 1. Use U to compute primitive variables, [rho, u, v, w, p]
// 2. Use U and primitive variables to compute fluxes, lambda
__global__ void compute_primitive_vars_fluxes(
    const double* __restrict__ const U_rho,     const double* __restrict__ const U_rho_u,   const double* __restrict__ const U_rho_v,   const double* __restrict__ const U_rho_w,   const double* __restrict__ const U_e,
          double* __restrict__ const lambda,
          double* __restrict__ const Fx_rho,          double* __restrict__ const Fx_rho_u,        double* __restrict__ const Fx_rho_v,        double* __restrict__ const Fx_rho_w,        double* __restrict__ const Fx_e,
          double* __restrict__ const Fy_rho,          double* __restrict__ const Fy_rho_u,        double* __restrict__ const Fy_rho_v,        double* __restrict__ const Fy_rho_w,        double* __restrict__ const Fy_e,
          double* __restrict__ const Fz_rho,          double* __restrict__ const Fz_rho_u,        double* __restrict__ const Fz_rho_v,        double* __restrict__ const Fz_rho_w,        double* __restrict__ const Fz_e
) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;

    if (i >= (NX+2*BUFFER) || j >= (NY+2*BUFFER) || k >= (NZ+2*BUFFER)) return;

    const int idx = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + i;

    // 1. Compute primitive variables
    const double W_rho = U_rho[idx];
    const double W_u   = U_rho_u[idx] / W_rho;
    const double W_v   = U_rho_v[idx] / W_rho;
    const double W_w   = U_rho_w[idx] / W_rho;
    const double W_p   = (GAMMA - 1.0) * (U_e[idx] - 0.5 * W_rho * (W_u * W_u + W_v * W_v + W_w * W_w));

    // 2. Compute fluxes Fx, Fy, Fz
    // x-direction flux (Fx)
    Fx_rho[idx]    = W_rho * W_u;
    Fx_rho_u[idx]  = W_rho * W_u * W_u + W_p;
    Fx_rho_v[idx]  = W_rho * W_u * W_v;
    Fx_rho_w[idx]  = W_rho * W_u * W_w;
    Fx_e[idx]      = (U_e[idx] + W_p) * W_u;

    // y-direction flux (Fy)
    Fy_rho[idx]    = W_rho * W_v;
    Fy_rho_u[idx]  = W_rho * W_v * W_u;
    Fy_rho_v[idx]  = W_rho * W_v * W_v + W_p;
    Fy_rho_w[idx]  = W_rho * W_v * W_w;
    Fy_e[idx]      = (U_e[idx] + W_p) * W_v;

    // z-direction flux (Fz)
    Fz_rho[idx]    = W_rho * W_w;
    Fz_rho_u[idx]  = W_rho * W_w * W_u;
    Fz_rho_v[idx]  = W_rho * W_w * W_v;
    Fz_rho_w[idx]  = W_rho * W_w * W_w + W_p;
    Fz_e[idx]      = (U_e[idx] + W_p) * W_w;

    // 3. Optional: Compute maximum eigenvalue (lambda) for CFL condition
    const double speed_of_sound = sqrt(GAMMA * W_p / W_rho);

    const double lambda_x = fabs(W_u) + speed_of_sound;
    const double lambda_y = fabs(W_v) + speed_of_sound;
    const double lambda_z = fabs(W_w) + speed_of_sound;

    lambda[idx] = fmax(fmax(lambda_x, lambda_y), lambda_z);

}

// Flux vector splitting
// F^+ = 0.5 * (F + lambda * U), F^- = 0.5 * (F - lambda * U)
__global__ void flux_vector_splitting(
    const double* __restrict__ const U,         const double* __restrict__ const lambda,
    const double* __restrict__ const Fx,              double* __restrict__ const Fx_plus,         double* __restrict__ const Fx_minus,
    const double* __restrict__ const Fy,              double* __restrict__ const Fy_plus,         double* __restrict__ const Fy_minus,
    const double* __restrict__ const Fz,              double* __restrict__ const Fz_plus,         double* __restrict__ const Fz_minus
) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;

    if (i >= (NX+2*BUFFER) || j >= (NY+2*BUFFER) || k >= (NZ+2*BUFFER)) return;

    const int idx = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + i;

    Fx_plus[idx] = 0.5 * (Fx[idx] + lambda[idx] * U[idx]);
    Fy_plus[idx] = 0.5 * (Fy[idx] + lambda[idx] * U[idx]);
    Fz_plus[idx] = 0.5 * (Fz[idx] + lambda[idx] * U[idx]);

    Fx_minus[idx] = 0.5 * (Fx[idx] - lambda[idx] * U[idx]);
    Fy_minus[idx] = 0.5 * (Fy[idx] - lambda[idx] * U[idx]);
    Fz_minus[idx] = 0.5 * (Fz[idx] - lambda[idx] * U[idx]);

}

// Reconstruct interface fluxes, F_{i+1/2} and F_{i-1/2}
// F_half_plus[i]  = f^{+}_{i+1/2}
// F_half_minus[i] = f^{-}_{i-1/2}
__global__ void reconstruct_interface_flux(
          double* __restrict__ const Fx_half_plus,        double* __restrict__ const Fx_half_minus,
          double* __restrict__ const Fy_half_plus,        double* __restrict__ const Fy_half_minus,
          double* __restrict__ const Fz_half_plus,        double* __restrict__ const Fz_half_minus,
    const double* __restrict__ const Fx_plus,       const double* __restrict__ const Fx_minus,
    const double* __restrict__ const Fy_plus,       const double* __restrict__ const Fy_minus,
    const double* __restrict__ const Fz_plus,       const double* __restrict__ const Fz_minus
) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;

    if (
        i < BUFFER - 1 || i >= NX + BUFFER + 1 ||
        j < BUFFER - 1 || j >= NY + BUFFER + 1 ||
        k < BUFFER - 1 || k >= NZ + BUFFER + 1
    ) return;

    const int idx = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + i;

    const int nFace = (NX+2*BUFFER)*(NZ+2*BUFFER);
    const int nLine = (NX+2*BUFFER);

    Fx_half_plus[idx] = MINMOD( Fx_plus[idx-1],     Fx_plus[idx], Fx_plus[idx+1]     );
    Fy_half_plus[idx] = MINMOD( Fy_plus[idx-nFace], Fy_plus[idx], Fy_plus[idx+nFace] );
    Fz_half_plus[idx] = MINMOD( Fz_plus[idx-nLine], Fz_plus[idx], Fz_plus[idx+nLine] );

    Fx_half_minus[idx] = MINMOD( Fx_minus[idx+1],     Fx_minus[idx], Fx_minus[idx-1]     );
    Fy_half_minus[idx] = MINMOD( Fy_minus[idx+nFace], Fy_minus[idx], Fy_minus[idx-nFace] );
    Fz_half_minus[idx] = MINMOD( Fz_minus[idx+nLine], Fz_minus[idx], Fz_minus[idx-nLine] );

}

__global__ void compute_RHS(
    const double* __restrict__ const Fx_half_plus,  const double* __restrict__ const Fy_half_plus,  const double* __restrict__ const Fz_half_plus,
    const double* __restrict__ const Fx_half_minus, const double* __restrict__ const Fy_half_minus, const double* __restrict__ const Fz_half_minus,
          double* __restrict__ const RHS
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

    const int nFace = (NX+2*BUFFER)*(NZ+2*BUFFER);
    const int nLine = (NX+2*BUFFER);

    const double dx = (X_MAX - X_MIN) / (double)(NX - 1);
    const double dy = (Y_MAX - Y_MIN) / (double)(NY - 1);
    const double dz = (Z_MAX - Z_MIN) / (double)(NZ - 1);

    const double Fx_half_front = Fx_half_plus[idx] + Fx_half_minus[idx+1];
    const double Fy_half_right = Fy_half_plus[idx] + Fy_half_minus[idx+nFace];
    const double Fz_half_up    = Fz_half_plus[idx] + Fz_half_minus[idx+nLine];

    const double Fx_half_back  = Fx_half_plus[idx-1]     + Fx_half_minus[idx];
    const double Fy_half_left  = Fy_half_plus[idx-nFace] + Fy_half_minus[idx];
    const double Fz_half_down  = Fz_half_plus[idx-nLine] + Fz_half_minus[idx];

    RHS[idx] = - (Fx_half_front - Fx_half_back) / dx - (Fy_half_right - Fy_half_left) / dy - (Fz_half_up - Fz_half_down) / dz;

}

void ComputeRHS(
          double* __restrict__ const RHS[5],
    const double* __restrict__ const U[5]
) {
    
    dim3 block(32, 1, 1);
    dim3 grid((NX+2*BUFFER)/32 + 1, (NY+2*BUFFER), (NZ+2*BUFFER));

    compute_primitive_vars_fluxes<<< grid, block >>>(
        U[0], U[1], U[2], U[3], U[4],
        lambda,
        Fx[0], Fx[1], Fx[2], Fx[3], Fx[4],
        Fy[0], Fy[1], Fy[2], Fy[3], Fy[4],
        Fz[0], Fz[1], Fz[2], Fz[3], Fz[4]
    );
    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaDeviceSynchronize() );

    for ( int i = 0; i < 5; i++ ){
        flux_vector_splitting<<< grid, block, 0, stream[i] >>>(
            U[i],  lambda,
            Fx[i], Fx_plus[i], Fx_minus[i],
            Fy[i], Fy_plus[i], Fy_minus[i],
            Fz[i], Fz_plus[i], Fz_minus[i]
        );
        CHECK_CUDA( cudaGetLastError() );
    }
    
    for ( int i = 0; i < 5; i++ ){
        reconstruct_interface_flux<<< grid, block, 0, stream[i] >>>(
            Fx_half_plus[i], Fx_half_minus[i],
            Fy_half_plus[i], Fy_half_minus[i],
            Fz_half_plus[i], Fz_half_minus[i],
            Fx_plus[i], Fx_minus[i],
            Fy_plus[i], Fy_minus[i],
            Fz_plus[i], Fz_minus[i]
        );
        CHECK_CUDA( cudaGetLastError() );
    }

    for ( int i = 0; i < 5; i++ ){
        compute_RHS<<< grid, block, 0, stream[i] >>>(
            Fx_half_plus[i],  Fy_half_plus[i],  Fz_half_plus[i],
            Fx_half_minus[i], Fy_half_minus[i], Fz_half_minus[i],
            RHS[i]
        );
        CHECK_CUDA( cudaGetLastError() );
    }

    for( int i = 0; i < 5; i++ ){
        CHECK_CUDA( cudaStreamSynchronize(stream[i]) );
    }

    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaDeviceSynchronize() );
}

__global__ void ApplyBoundaryConditions_xdir(
    double* __restrict__ const U
) {

    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;

    if ( j >= NY+2*BUFFER || k >= NZ+2*BUFFER ) return;

    int idx_0 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + 0;
    int idx_1 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + 1;
    int idx_2 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + 2;

    int idx_3 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + 3;
    int idx_4 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + 4;
    int idx_5 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + 5;

    // zero-order extrapolation
    // U[idx_0] = U[idx_3];
    // U[idx_1] = U[idx_3];
    // U[idx_2] = U[idx_3];

    // second-order extrapolation
    U[idx_0] = 10.0 * U[idx_3] - 15.0 * U[idx_4] + 6.0 * U[idx_5];
    U[idx_1] = 6.0  * U[idx_3] - 8.0  * U[idx_4] + 3.0 * U[idx_5];
    U[idx_2] = 3.0  * U[idx_3] - 3.0  * U[idx_4] + U[idx_5];

    idx_0 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + (NX+2*BUFFER-1);
    idx_1 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + (NX+2*BUFFER-2);
    idx_2 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + (NX+2*BUFFER-3);

    idx_3 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + (NX+2*BUFFER-4);
    idx_4 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + (NX+2*BUFFER-5);
    idx_5 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + (NX+2*BUFFER-6);

    // zero-order extrapolation
    // U[idx_0] = U[idx_3];
    // U[idx_1] = U[idx_3];
    // U[idx_2] = U[idx_3];

    // second-order extrapolation
    U[idx_0] = 10.0 * U[idx_3] - 15.0 * U[idx_4] + 6.0 * U[idx_5];
    U[idx_1] = 6.0  * U[idx_3] - 8.0  * U[idx_4] + 3.0 * U[idx_5];
    U[idx_2] = 3.0  * U[idx_3] - 3.0  * U[idx_4] + U[idx_5];
}

__global__ void ApplyBoundaryConditions_ydir(
    double* __restrict__ const U
) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;

    if ( i >= NX+2*BUFFER || k >= NZ+2*BUFFER ) return;

    int idx_0 = 0*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + i;
    int idx_1 = 1*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + i;
    int idx_2 = 2*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + i;

    int idx_3 = 3*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + i;
    int idx_4 = 4*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + i;
    int idx_5 = 5*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + i;

    // zero-order extrapolation
    // U[idx_0] = U[idx_3];
    // U[idx_1] = U[idx_3];
    // U[idx_2] = U[idx_3];

    // second-order extrapolation
    U[idx_0] = 10.0 * U[idx_3] - 15.0 * U[idx_4] + 6.0 * U[idx_5];
    U[idx_1] = 6.0  * U[idx_3] - 8.0  * U[idx_4] + 3.0 * U[idx_5];
    U[idx_2] = 3.0  * U[idx_3] - 3.0  * U[idx_4] + U[idx_5];

    idx_0 = (NY+2*BUFFER-1)*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + i;
    idx_1 = (NY+2*BUFFER-2)*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + i;
    idx_2 = (NY+2*BUFFER-3)*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + i;

    idx_3 = (NY+2*BUFFER-4)*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + i;
    idx_4 = (NY+2*BUFFER-5)*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + i;
    idx_5 = (NY+2*BUFFER-6)*(NX+2*BUFFER)*(NZ+2*BUFFER) + k*(NX+2*BUFFER) + i;

    // zero-order extrapolation
    // U[idx_0] = U[idx_3];
    // U[idx_1] = U[idx_3];
    // U[idx_2] = U[idx_3];

    // second-order extrapolation
    U[idx_0] = 10.0 * U[idx_3] - 15.0 * U[idx_4] + 6.0 * U[idx_5];
    U[idx_1] = 6.0  * U[idx_3] - 8.0  * U[idx_4] + 3.0 * U[idx_5];
    U[idx_2] = 3.0  * U[idx_3] - 3.0  * U[idx_4] + U[idx_5];
}

__global__ void ApplyBoundaryConditions_zdir(
    double* __restrict__ const U
) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    if ( i >= NX+2*BUFFER || j >= NY+2*BUFFER ) return;

    int idx_0 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + 0*(NX+2*BUFFER) + i;
    int idx_1 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + 1*(NX+2*BUFFER) + i;
    int idx_2 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + 2*(NX+2*BUFFER) + i;

    int idx_3 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + 3*(NX+2*BUFFER) + i;
    int idx_4 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + 4*(NX+2*BUFFER) + i;
    int idx_5 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + 5*(NX+2*BUFFER) + i;

    // zero-order extrapolation
    // U[idx_0] = U[idx_3];
    // U[idx_1] = U[idx_3];
    // U[idx_2] = U[idx_3];

    // second-order extrapolation
    U[idx_0] = 10.0 * U[idx_3] - 15.0 * U[idx_4] + 6.0 * U[idx_5];
    U[idx_1] = 6.0  * U[idx_3] - 8.0  * U[idx_4] + 3.0 * U[idx_5];
    U[idx_2] = 3.0  * U[idx_3] - 3.0  * U[idx_4] + U[idx_5];

    idx_0 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + (NZ+2*BUFFER-1)*(NX+2*BUFFER) + i;
    idx_1 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + (NZ+2*BUFFER-2)*(NX+2*BUFFER) + i;
    idx_2 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + (NZ+2*BUFFER-3)*(NX+2*BUFFER) + i;

    idx_3 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + (NZ+2*BUFFER-4)*(NX+2*BUFFER) + i;
    idx_4 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + (NZ+2*BUFFER-5)*(NX+2*BUFFER) + i;
    idx_5 = j*(NX+2*BUFFER)*(NZ+2*BUFFER) + (NZ+2*BUFFER-6)*(NX+2*BUFFER) + i;

    // zero-order extrapolation
    // U[idx_0] = U[idx_3];
    // U[idx_1] = U[idx_3];
    // U[idx_2] = U[idx_3];

    // second-order extrapolation
    U[idx_0] = 10.0 * U[idx_3] - 15.0 * U[idx_4] + 6.0 * U[idx_5];
    U[idx_1] = 6.0  * U[idx_3] - 8.0  * U[idx_4] + 3.0 * U[idx_5];
    U[idx_2] = 3.0  * U[idx_3] - 3.0  * U[idx_4] + U[idx_5];
}

#endif