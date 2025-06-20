#ifndef MEMORY_MANAGE_H
#define MEMORY_MANAGE_H

void AllocateDeviceArray(const size_t nBytes, const int num_arrays, ...) {
	va_list args;
	va_start( args, num_arrays );

	for( int i = 0; i < num_arrays; i++ ) {
        double **tmp = va_arg(args, double**);
		CHECK_CUDA( cudaMalloc( (void**)tmp, nBytes) );
	}

	va_end( args );
}

void FreeDeviceArray(const int num_arrays, ...) {
    va_list args;
    va_start( args, num_arrays );

    for( int  i = 0; i < num_arrays; i++ ) {
        CHECK_CUDA( cudaFree( (void*)(va_arg(args, double*)) ) );
    }

    va_end( args );
}

void AllocateMemory() {

    CHECK_CUDA( cudaMallocHost( (void**)&x_host, (NX + 2 * BUFFER) * sizeof(double) ) );
    CHECK_CUDA( cudaMallocHost( (void**)&y_host, (NY + 2 * BUFFER) * sizeof(double) ) );
    CHECK_CUDA( cudaMallocHost( (void**)&z_host, (NZ + 2 * BUFFER) * sizeof(double) ) );

    CHECK_CUDA( cudaMalloc( (void**)&x, (NX+2*BUFFER) * sizeof(double) ) );
    CHECK_CUDA( cudaMalloc( (void**)&y, (NY+2*BUFFER) * sizeof(double) ) );
    CHECK_CUDA( cudaMalloc( (void**)&z, (NZ+2*BUFFER) * sizeof(double) ) );

    size_t nBytes;

    nBytes = (NX+2*BUFFER) * (NY+2*BUFFER) * (NZ+2*BUFFER) * sizeof(double);

    AllocateDeviceArray(nBytes, 1, &lambda);

    for ( int i = 0; i < 5; i++ ) {
        AllocateDeviceArray(nBytes, 4, &U[i], &U1[i], &U2[i], &RHS[i]);
        AllocateDeviceArray(nBytes, 1, &W[i]);
        AllocateDeviceArray(nBytes, 5, &Fx[i], &Fx_plus[i], &Fx_minus[i], &Fx_half_plus[i], &Fx_half_minus[i]);
        AllocateDeviceArray(nBytes, 5, &Fy[i], &Fy_plus[i], &Fy_minus[i], &Fy_half_plus[i], &Fy_half_minus[i]);
        AllocateDeviceArray(nBytes, 5, &Fz[i], &Fz_plus[i], &Fz_minus[i], &Fz_half_plus[i], &Fz_half_minus[i]);
    }

    for( int i = 0; i < 5; i++ ) CHECK_CUDA( cudaStreamCreate( &stream[i] ) );

    CHECK_CUDA( cudaEventCreate( &start1 ) );
    CHECK_CUDA( cudaEventCreate( &stop1  ) );

}

void FreeMemory() {

    FreeDeviceArray(1, lambda);
    FreeDeviceArray(3, x, y, z);
    
    for ( int i = 0; i < 5; i++ ) {
        FreeDeviceArray(4, U[i], U1[i], U2[i], RHS[i]);
        FreeDeviceArray(1, W[i]);
        FreeDeviceArray(5, Fx[i], Fx_plus[i], Fx_minus[i], Fx_half_plus[i], Fx_half_minus[i]);
        FreeDeviceArray(5, Fy[i], Fy_plus[i], Fy_minus[i], Fy_half_plus[i], Fy_half_minus[i]);
        FreeDeviceArray(5, Fz[i], Fz_plus[i], Fz_minus[i], Fz_half_plus[i], Fz_half_minus[i]);
    }

    for( int i = 0; i < 5; i++ ) CHECK_CUDA( cudaStreamDestroy( stream[i] ) );

    // Free the host memory
    CHECK_CUDA( cudaFreeHost(x_host) );
    CHECK_CUDA( cudaFreeHost(y_host) );
    CHECK_CUDA( cudaFreeHost(z_host) );

    CHECK_CUDA( cudaEventDestroy( start1 ) );
    CHECK_CUDA( cudaEventDestroy( stop1  ) );

}

#endif