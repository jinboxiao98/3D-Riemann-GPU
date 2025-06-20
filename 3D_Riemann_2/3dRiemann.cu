#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <stdarg.h>
#include <time.h>

#define         NX          801
#define         NY          801
#define         NZ          3
#define         BUFFER      3

#define         GAMMA       1.4

#define         X_MAX       1.0
#define         X_MIN       0.0

#define         Y_MAX       1.0
#define         Y_MIN       0.0

#define         Z_MAX       0.1
#define         Z_MIN       0.0

#define         DT          0.00001
#define         FINAL_TIME  0.3

#define         TIME_FLAG   1000    

double *x, *y, *z;

double *x_host = NULL;
double *y_host = NULL;
double *z_host = NULL;

/***************** 3rd-order Runge Kutta variables *****************/
// 0: rho
// 1: rho * u 
// 2: rho * v
// 3: rho * w
// 4: E
double *U[5], *U1[5], *U2[5];
double *RHS[5];

/***************** 3rd-order Runge Kutta temporary variables *****************/
// 0: rho
// 1: u
// 2: v
// 3: w
// 4: p
double *W[5];

//    x:                    y:                  z:
// 0: rho * u               rho * v             rho * w
// 1: rho * u^2 + p         rho * v * u         rho * w * u
// 2: rho * u * v           rho * v^2 + p       rho * w * v
// 3: rho * u * w           rho * v * w         rho * w^2 + p
// 4: (E + p) * u           (E + p) * v         (E + p) * w
double *Fx[5], *Fx_plus[5], *Fx_minus[5], *Fx_half_plus[5], *Fx_half_minus[5];
double *Fy[5], *Fy_plus[5], *Fy_minus[5], *Fy_half_plus[5], *Fy_half_minus[5];
double *Fz[5], *Fz_plus[5], *Fz_minus[5], *Fz_half_plus[5], *Fz_half_minus[5];

double *lambda;

/***************** FileIO *****************/
double* h_U_rho;
double* h_U_rho_u;
double* h_U_rho_v;
double* h_U_rho_w;
double* h_U_e;

double* h_rho;
double* h_u;
double* h_v;
double* h_w;
double* h_p;
double* h_x;
double* h_y;
double* h_z;

cudaStream_t  stream[5];

cudaEvent_t   start1,  stop1;

#include "common.h"
#include "memoryManage.h"
#include "Riemann.h"
#include "RungeKutta.h"
#include "fileIO.h"

int main(int argc, char** argv) {
    AllocateMemory();

    InitialCondition();

    dim3 block(32, 1, 1);
    dim3 grid((NX+2*BUFFER)/32 + 1, (NY+2*BUFFER), (NZ+2*BUFFER));

    dim3 blockBdry_x(1, 1, 32 );
    dim3 gridBdry_x(1,  (NY+2*BUFFER), (NZ+2*BUFFER)/32 + 1);

    dim3 blockBdry_y(1, 1, 32);
    dim3 gridBdry_y((NX+2*BUFFER), 1, (NZ+2*BUFFER)/32 + 1);

    dim3 blockBdry_z(32, 1, 1);
    dim3 gridBdry_z((NX+2*BUFFER)/32 + 1, (NY+2*BUFFER), 1);

    printf("Running 3D Riemann problem with %d x %d x %d grid points\n", NX, NY, NZ);

	CHECK_CUDA( cudaEventRecord(start1,0) );

    int step = 0;
    for( double time = DT; time < FINAL_TIME; time+= DT, step++ ) {

        // 3rd order Runge-Kutta time-stepping
        ComputeRHS(RHS, U);
        for ( int i = 0; i < 5; i++ ) {
            RungeKuttaKernel_1<<< grid, block, 0, stream[i] >>>(
                U1[i], U[i], RHS[i]
            );
        }
        for ( int i = 0; i < 5; i++ ) { ApplyBoundaryConditions_xdir<<< gridBdry_x, blockBdry_x, 0, stream[i] >>>( U1[i] ); }
        for ( int i = 0; i < 5; i++ ) { ApplyBoundaryConditions_ydir<<< gridBdry_y, blockBdry_y, 0, stream[i] >>>( U1[i] ); }
        for ( int i = 0; i < 5; i++ ) { ApplyBoundaryConditions_zdir<<< gridBdry_z, blockBdry_z, 0, stream[i] >>>( U1[i] ); }

        for( int i = 0; i < 5; i++ ) CHECK_CUDA( cudaStreamSynchronize(stream[i]) );

        ComputeRHS(RHS, U1);
        for ( int i = 0; i < 5; i++ ) {
            RungeKuttaKernel_2<<< grid, block, 0, stream[i] >>>(
                U2[i], U[i], U1[i], RHS[i]
            );
        }
        for ( int i = 0; i < 5; i++ ) { ApplyBoundaryConditions_xdir<<< gridBdry_x, blockBdry_x, 0, stream[i] >>>( U2[i] ); }
        for ( int i = 0; i < 5; i++ ) { ApplyBoundaryConditions_ydir<<< gridBdry_y, blockBdry_y, 0, stream[i] >>>( U2[i] ); }
        for ( int i = 0; i < 5; i++ ) { ApplyBoundaryConditions_zdir<<< gridBdry_z, blockBdry_z, 0, stream[i] >>>( U2[i] ); }

        for( int i = 0; i < 5; i++ ) CHECK_CUDA( cudaStreamSynchronize(stream[i]) );

        ComputeRHS(RHS, U2);
        for ( int i = 0; i < 5; i++ ) {
            RungeKuttaKernel_3<<< grid, block, 0, stream[i] >>>(
                U[i], U2[i], RHS[i]
            );
        }
        for ( int i = 0; i < 5; i++ ) { ApplyBoundaryConditions_xdir<<< gridBdry_x, blockBdry_x, 0, stream[i] >>>( U[i] ); }
        for ( int i = 0; i < 5; i++ ) { ApplyBoundaryConditions_ydir<<< gridBdry_y, blockBdry_y, 0, stream[i] >>>( U[i] ); }
        for ( int i = 0; i < 5; i++ ) { ApplyBoundaryConditions_zdir<<< gridBdry_z, blockBdry_z, 0, stream[i] >>>( U[i] ); }

        for( int i = 0; i < 5; i++ ) CHECK_CUDA( cudaStreamSynchronize(stream[i]) );

        if ( step % TIME_FLAG == 0 ) {
            CHECK_CUDA( cudaEventRecord( stop1,0 ) );
            CHECK_CUDA( cudaEventSynchronize( stop1 ) );
            float cuda_time;
            CHECK_CUDA( cudaEventElapsedTime( &cuda_time,start1,stop1 ) );

            printf("+----------------------------------------------------------------+\n");
            printf("| Current time is %.5f, Step = %d \n",time, step);
            printf("| %s running with %4dx%4dx%4d grids            \n", argv[0], (int)NX, (int)NY, (int)NZ );
            printf("| Running %6f mins                                           \n", (cuda_time/60/1000) );
            printf("+----------------------------------------------------------------+\n");

            CHECK_CUDA( cudaEventRecord(start1,0) );
        }
    }

    printf("Final time: %f\n", FINAL_TIME);

    OutputResult();

    FreeMemory();

    return 0;
}
