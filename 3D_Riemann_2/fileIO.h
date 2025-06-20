#ifndef FILE_IO_H
#define FILE_IO_H

__global__ void Launch_InitialCondition(
    double* __restrict__ const U_rho, 
    double* __restrict__ const U_rho_u, 
    double* __restrict__ const U_rho_v, 
    double* __restrict__ const U_rho_w, 
    double* __restrict__ const U_e
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= NX + 2 * BUFFER || j >= NY + 2 * BUFFER || k >= NZ + 2 * BUFFER) return;

    const int idx = j * (NX + 2 * BUFFER) * (NZ + 2 * BUFFER) + k * (NX + 2 * BUFFER) + i;

    const double x = (double)(i - BUFFER) * (X_MAX - X_MIN) / (double)(NX - 1) + X_MIN;
    const double y = (double)(j - BUFFER) * (Y_MAX - Y_MIN) / (double)(NY - 1) + Y_MIN;
    const double z = (double)(k - BUFFER) * (Z_MAX - Z_MIN) / (double)(NZ - 1) + Z_MIN;

    // Set initial conditions
    double rho, u, v, w, p;

    w = 0.0;

    if (x < 0.5 && y < 0.5) {
        rho = 1.0;
        u   = -0.75;
        v   = 0.5;
        p   = 1.0;
    }
    // Quadrant II: 0 ≤ x ≤ 0.5, 0.5 ≤ y ≤ 1
    if (x < 0.5 && y >= 0.5) {
        rho = 2.0;
        u   = 0.75;
        v   = 0.5;
        p   = 1.0;
    }
    // Quadrant III: 0.5 < x ≤ 1, 0 ≤ y ≤ 0.5
    if (x >= 0.5 && y < 0.5) {
        rho = 3.0;
        u   = -0.75;
        v   = -0.5;
        p   = 1.0;
    }
    // Quadrant IV: 0.5 < x ≤ 1, 0.5 < y ≤ 1
    if (x >= 0.5 && y >= 0.5) {
        rho = 1.0;
        u   = 0.75;
        v   = -0.5;
        p   = 1.0;
    }
    /* if (x < 0.5) {
        rho = 1.0;
        u = 0.0;
        v = 0.0;
        w = 0.0;
        p = 1.0;
    } else {
        rho = 0.125;
        u = 0.0;
        v = 0.0;
        w = 0.0;
        p = 0.1;
    } */

    U_rho[idx]     = rho;
    U_rho_u[idx]   = rho * u;
    U_rho_v[idx]   = rho * v;
    U_rho_w[idx]   = rho * w;
    U_e[idx]       = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v + w * w);
}

void InitialCondition() {

    // Initial x, y, z coordinates
    for ( int i = 0; i < NX + 2 * BUFFER; i++ ) {
        x_host[i] = X_MIN + (double)(i - BUFFER) * (X_MAX - X_MIN) / (double)(NX - 1);
    }
    for ( int i = 0; i < NY + 2 * BUFFER; i++ ) {
        y_host[i] = Y_MIN + (double)(i - BUFFER) * (Y_MAX - Y_MIN) / (double)(NY - 1);
    }
    for ( int i = 0; i < NZ + 2 * BUFFER; i++ ) {
        z_host[i] = Z_MIN + (double)(i - BUFFER) * (Z_MAX - Z_MIN) / (double)(NZ - 1);
    }

    CHECK_CUDA( cudaMemcpy( x, x_host, (NX + 2 * BUFFER) * sizeof(double), cudaMemcpyHostToDevice ) );
    CHECK_CUDA( cudaMemcpy( y, y_host, (NY + 2 * BUFFER) * sizeof(double), cudaMemcpyHostToDevice ) );
    CHECK_CUDA( cudaMemcpy( z, z_host, (NZ + 2 * BUFFER) * sizeof(double), cudaMemcpyHostToDevice ) );

    dim3 block(32, 1, 1);
    dim3 grid((NX+2*BUFFER)/32 + 1, (NY+2*BUFFER), (NZ+2*BUFFER));

    printf("Setting initial conditions...\n");

    Launch_InitialCondition<<< grid, block >>>(
        U[0], U[1], U[2], U[3], U[4]
    );

    printf("Initial conditions set.\n");

    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaDeviceSynchronize() );

    dim3 blockBdry_x(1, 1, 32 );
    dim3 gridBdry_x(1,  (NY+2*BUFFER), (NZ+2*BUFFER)/32 + 1);

    dim3 blockBdry_y(1, 1, 32);
    dim3 gridBdry_y((NX+2*BUFFER), 1, (NZ+2*BUFFER)/32 + 1);

    dim3 blockBdry_z(32, 1, 1);
    dim3 gridBdry_z((NX+2*BUFFER)/32 + 1, (NY+2*BUFFER), 1);

    for ( int i = 0; i < 5; i++ ) {

        ApplyBoundaryConditions_xdir<<< gridBdry_x, blockBdry_x >>>( U[i] );
        ApplyBoundaryConditions_ydir<<< gridBdry_y, blockBdry_y >>>( U[i] );
        ApplyBoundaryConditions_zdir<<< gridBdry_z, blockBdry_z >>>( U[i] );

    }

    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaDeviceSynchronize() );
    
}

void WriteBinaryFile(const char* filename, const double* data, size_t size) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: cannot open file %s for writing.\n", filename);
        return;
    }
    fwrite(data, sizeof(double), size, file);
    fclose(file);
}

void wirte_ASCII_of_str(char * str, FILE * file)
{
    int value = 0;

    while ((*str) != '\0'){
        value = (int)*str;
        fwrite(&value, sizeof(int), 1, file);
        str++;
    }

    char null_char[] = "";

    value = (int)*null_char;

    fwrite(&value, sizeof(int), 1, file);
}

void OutputTecplotPltFile(){

    const int NX_TOT = NX + 2 * BUFFER;
    const int NY_TOT = NY + 2 * BUFFER;
    const int NZ_TOT = NZ + 2 * BUFFER;

    char filename_E2[300];

    sprintf(filename_E2, "final_result.plt");
    printf("%s\n", filename_E2);

    FILE *fpE3;

    fpE3 = fopen(filename_E2, "wb");

    int IMax = NX;
    int JMax = NY;
    int KMax = NZ;

    char Title[] = "Particle intensity";

    char Varname1[] = "X";

    char Varname2[] = "Y";

    char Varname3[] = "Z";

    char Varname4[] = "Rho";
    
    char Varname5[] = "U";
	
	char Varname6[] = "V"; 
	
	char Varname7[] = "W";

    char Varname8[] = "P";

    char Zonename1[] = "Zone 001";

    float ZONEMARKER = 299.0;

    float EOHMARKER = 357.0;

    char MagicNumber[] = "#!TDV101";
    fwrite(MagicNumber, 8, 1, fpE3);

    int IntegerValue = 1;
    fwrite(&IntegerValue, sizeof(IntegerValue), 1, fpE3);

    wirte_ASCII_of_str(Title, fpE3);

    int NumVar = 8;
    fwrite(&NumVar, sizeof(NumVar), 1, fpE3);

    wirte_ASCII_of_str(Varname1, fpE3);
    wirte_ASCII_of_str(Varname2, fpE3);
    wirte_ASCII_of_str(Varname3, fpE3);
    wirte_ASCII_of_str(Varname4, fpE3);
    wirte_ASCII_of_str(Varname5, fpE3);
    wirte_ASCII_of_str(Varname6, fpE3);
	wirte_ASCII_of_str(Varname7, fpE3);
    wirte_ASCII_of_str(Varname8, fpE3);

    fwrite(&ZONEMARKER, 1, sizeof(ZONEMARKER), fpE3);

    wirte_ASCII_of_str(Zonename1, fpE3);

    int ZoneColor = -1;
    fwrite(&ZoneColor, sizeof(ZoneColor), 1, fpE3);

    int ZoneType = 0;
    fwrite(&ZoneType, sizeof(ZoneType), 1, fpE3);

    int DaraPacking = 1;
    fwrite(&DaraPacking, sizeof(DaraPacking), 1, fpE3);

    int SpecifyVarLocation = 0;
    fwrite(&SpecifyVarLocation, sizeof(SpecifyVarLocation), 1, fpE3);

    int NumOfNeighbor = 0;
    fwrite(&NumOfNeighbor, sizeof(NumOfNeighbor), 1, fpE3);

    fwrite(&IMax, sizeof(IMax), 1, fpE3);
    fwrite(&JMax, sizeof(JMax), 1, fpE3);
    fwrite(&KMax, sizeof(KMax), 1, fpE3);

    int AuxiliaryName = 0;
    fwrite(&AuxiliaryName, sizeof(AuxiliaryName), 1, fpE3);


    fwrite(&EOHMARKER, sizeof(EOHMARKER), 1, fpE3);

    fwrite(&ZONEMARKER, sizeof(ZONEMARKER), 1, fpE3);

    //--------variable c_str format, 1 = Float, 2 = Double, 3 = LongInt, 4 = ShortInt, 5 = Byte, 6 = Bit
    int fomat1 = 2;
    int fomat2 = 2;
    int fomat3 = 2;
    int fomat4 = 2;
    int fomat5 = 2;
    int fomat6 = 2;
	int fomat7 = 2;
    int fomat8 = 2;
    fwrite(&fomat1, sizeof(fomat1), 1, fpE3);
    fwrite(&fomat2, sizeof(fomat2), 1, fpE3);
    fwrite(&fomat3, sizeof(fomat3), 1, fpE3);
    fwrite(&fomat4, sizeof(fomat4), 1, fpE3);
    fwrite(&fomat5, sizeof(fomat5), 1, fpE3);
    fwrite(&fomat6, sizeof(fomat6), 1, fpE3);
	fwrite(&fomat7, sizeof(fomat7), 1, fpE3);
    fwrite(&fomat8, sizeof(fomat8), 1, fpE3);

    int HasVarSharing = 0;
    fwrite(&HasVarSharing, sizeof(HasVarSharing), 1, fpE3);


    int ZoneNumToShareConnectivity = -1;
    fwrite(&ZoneNumToShareConnectivity, sizeof(ZoneNumToShareConnectivity), 1, fpE3);


    for ( int k = 0; k < NZ; k++ ) {
    for ( int j = 0; j < NY; j++ ) {
    for ( int i = 0; i < NX; i++ ) {

        int idx = j*NZ*NX + k*NX + i;

        // Write coordinates and conservative variables
	    double x_d = x_host[i+BUFFER];
        double y_d = y_host[j+BUFFER];
        double z_d = z_host[k+BUFFER];

        double rho_d = h_rho[idx];
        double u_d   = h_u[idx];
        double v_d   = h_v[idx];
        double w_d   = h_w[idx];
        double p_d   = h_p[idx];

        fwrite(&x_d,   sizeof(double), 1, fpE3);
        fwrite(&y_d,   sizeof(double), 1, fpE3);
        fwrite(&z_d,   sizeof(double), 1, fpE3);
        fwrite(&rho_d, sizeof(double), 1, fpE3);
        fwrite(&u_d,   sizeof(double), 1, fpE3);
        fwrite(&v_d,   sizeof(double), 1, fpE3);
        fwrite(&w_d,   sizeof(double), 1, fpE3);
        fwrite(&p_d,   sizeof(double), 1, fpE3);
    }}}

    fclose(fpE3);

}


void OutputResult() {
    const int NX_TOT = NX + 2 * BUFFER;
    const int NY_TOT = NY + 2 * BUFFER;
    const int NZ_TOT = NZ + 2 * BUFFER;

    const int totalSize = NX_TOT * NY_TOT * NZ_TOT;
    const int outputSize = NX * NY * NZ;

    // Allocate full host arrays for conservative vars
    h_U_rho   = (double*)malloc(sizeof(double) * totalSize);
    h_U_rho_u = (double*)malloc(sizeof(double) * totalSize);
    h_U_rho_v = (double*)malloc(sizeof(double) * totalSize);
    h_U_rho_w = (double*)malloc(sizeof(double) * totalSize);
    h_U_e     = (double*)malloc(sizeof(double) * totalSize);

    // Allocate trimmed output arrays (interior only)
    h_rho = (double*)malloc(sizeof(double) * outputSize);
    h_u   = (double*)malloc(sizeof(double) * outputSize);
    h_v   = (double*)malloc(sizeof(double) * outputSize);
    h_w   = (double*)malloc(sizeof(double) * outputSize);
    h_p   = (double*)malloc(sizeof(double) * outputSize);
    h_x   = (double*)malloc(sizeof(double) * outputSize);
    h_y   = (double*)malloc(sizeof(double) * outputSize);
    h_z   = (double*)malloc(sizeof(double) * outputSize);

    // Copy from GPU to host
    CHECK_CUDA(cudaMemcpy(h_U_rho,   U[0], sizeof(double) * totalSize, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_U_rho_u, U[1], sizeof(double) * totalSize, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_U_rho_v, U[2], sizeof(double) * totalSize, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_U_rho_w, U[3], sizeof(double) * totalSize, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_U_e,     U[4], sizeof(double) * totalSize, cudaMemcpyDeviceToHost));

    // Compute primitive vars inside domain (excluding BUFFER)
    for (int j = BUFFER; j < NY + BUFFER; ++j) {
    for (int k = BUFFER; k < NZ + BUFFER; ++k) {
    for (int i = BUFFER; i < NX + BUFFER; ++i) {
        int idx = j * NZ_TOT * NX_TOT + k * NX_TOT + i;
        int out_idx = (j - BUFFER) * NX * NZ + (k - BUFFER) * NX + (i - BUFFER);

        double rho = h_U_rho[idx];
        double u   = h_U_rho_u[idx] / rho;
        double v   = h_U_rho_v[idx] / rho;
        double w   = h_U_rho_w[idx] / rho;
        double p   = (GAMMA - 1.0) * (h_U_e[idx] - 0.5 * rho * (u*u + v*v + w*w));

        h_x[out_idx] = x_host[i];
        h_y[out_idx] = y_host[j];
        h_z[out_idx] = z_host[k];

        h_rho[out_idx] = rho;
        h_u[out_idx]   = u;
        h_v[out_idx]   = v;
        h_w[out_idx]   = w;
        h_p[out_idx]   = p;

    }}}

    // Output files (interior only)
    WriteBinaryFile("rho.bin", h_rho, outputSize);
    WriteBinaryFile("u.bin",   h_u,   outputSize);
    WriteBinaryFile("v.bin",   h_v,   outputSize);
    WriteBinaryFile("w.bin",   h_w,   outputSize);
    WriteBinaryFile("p.bin",   h_p,   outputSize);
    WriteBinaryFile("x.bin",   h_x,   outputSize);
    WriteBinaryFile("y.bin",   h_y,   outputSize);
    WriteBinaryFile("z.bin",   h_z,   outputSize);

    printf("Interior data written (no BUFFER): %dx%dx%d grid\n", NX, NY, NZ);

    OutputTecplotPltFile();

    // Clean up
    free(h_U_rho); free(h_U_rho_u); free(h_U_rho_v); free(h_U_rho_w); free(h_U_e);
    free(h_rho); free(h_u); free(h_v); free(h_w); free(h_p);
    free(h_x); free(h_y); free(h_z);
}

#endif // FILE_IO_H