# 3D Riemann code fully based on GPU

This code solves the 3D Euler equations with freestream boundary conditions. The temporal discretization uses a 3rd-order Runge-Kutta scheme, while the spatial discretization employs a 2nd-order MUSCL scheme (with Minmod limiter).

## Compilation

If you are using a personal computer with an NVIDIA GeForce RTX 4090 GPU, you can compile the code as follows:
```bash
nvcc -arch sm_89 -o case_number_and_mesh_number 3DRiemann.cu
```
If you are using PACE, the batch file, ```riemann.sh```, is
```bash
#!/bin/bash
#SBATCH -J Riemann
#SBATCH -A gts-vyang6-coda20
#SBATCH -N1 --gres=gpu:1 -C A100-40GB
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -o Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjin60@gatech.edu

cd $SLURM_SUBMIT_DIR
module load gcc/12.3.0
module load cuda/12.1.1
module load mvapich2/2.3.7-1

cd /storage/home/hcoda1/9/bjin60/p-vyang6-0/Riemann

nvcc -arch sm_80 -o case_number_and_mesh_number 3dRiemann.cu

srun -n 1 ./case_number_and_mesh_number
```
The batch file can be executed by
```bash
sbatch riemann.sh
```


The gcc version is 11.4.0 on my personal computer, which can be checked by ```gcc --version```.

The cuda version is 12.9 on my personal computer, which can be checked by ```nvcc --version```.

CUDA can be installed by 
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_575.51.03_linux.run
sudo sh cuda_12.9.0_575.51.03_linux.ru
```

## Parameters

```C
// Mesh number in x, y, and z-directions
// Mesh number of ghost cells is default to be 3
#define         NX          401
#define         NY          401
#define         NZ          401
#define         BUFFER      3

#define         GAMMA       1.4

// Define the domain region
#define         X_MAX       1.0
#define         X_MIN       0.0

#define         Y_MAX       1.0
#define         Y_MIN       0.0

#define         Z_MAX       0.1
#define         Z_MIN       0.0

// Define the time step size and simulation end time
#define         DT          0.00001
#define         FINAL_TIME  0.3

// Frequency for checkpointing
#define         TIME_FLAG   1000  
```

## License

[MIT](https://choosealicense.com/licenses/mit/)

---

# 3D Euler Equations Solver (Finite-Volume Method with RK3-TVD)

This code solves the **3D compressible Euler equations** for inviscid gas dynamics:

\[
\frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \mathbf{F} = 0,
\]

where:
- \(\mathbf{U}\) is the vector of conserved variables,
- \(\mathbf{F} = (\mathbf{F}_x, \mathbf{F}_y, \mathbf{F}_z)\) are the flux vectors in \(x\), \(y\), and \(z\) directions.

---

## **Governing Equations**
### **Conserved Variables**
\[
\mathbf{U} = \begin{bmatrix}
\rho \\
\rho u \\
\rho v \\
\rho w \\
E
\end{bmatrix}, \quad
E = \rho e + \frac{1}{2}\rho (u^2 + v^2 + w^2)
\]

### **Flux Vectors**
\[
\mathbf{F}_x = \begin{bmatrix}
\rho u \\
\rho u^2 + p \\
\rho u v \\
\rho u w \\
u (E + p)
\end{bmatrix}, \quad
\mathbf{F}_y = \begin{bmatrix}
\rho v \\
\rho v u \\
\rho v^2 + p \\
\rho v w \\
v (E + p)
\end{bmatrix}, \quad
\mathbf{F}_z = \begin{bmatrix}
\rho w \\
\rho w u \\
\rho w v \\
\rho w^2 + p \\
w (E + p)
\end{bmatrix}
\]

- \(\rho\): Density  
- \(u, v, w\): Velocity components  
- \(p\): Pressure (from equation of state: \(p = (\gamma - 1) \left[ E - \frac{1}{2}\rho (u^2 + v^2 + w^2) \right]\))  
- \(\gamma = 1.4\): Ratio of specific heats  

---

## **Numerical Method**  
1. **Temporal Integration**: 3rd-order Runge-Kutta (RK3):  
   \[
   \begin{aligned}
   \mathbf{U}^{(1)} &= \mathbf{U}^n - \Delta t \, \nabla \cdot \mathbf{F}(\mathbf{U}^n), \\
   \mathbf{U}^{(2)} &= \frac{3}{4}\mathbf{U}^n + \frac{1}{4}\mathbf{U}^{(1)} - \frac{1}{4}\Delta t \, \nabla \cdot \mathbf{F}(\mathbf{U}^{(1)}), \\
   \mathbf{U}^{n+1} &= \frac{1}{3}\mathbf{U}^n + \frac{2}{3}\mathbf{U}^{(2)} - \frac{2}{3}\Delta t \, \nabla \cdot \mathbf{F}(\mathbf{U}^{(2)}).
   \end{aligned}
   \]

2. **Spatial Discretization**: Finite-volume method with local Lax-Friedrichs (LLF) flux splitting.

    ```C
    __global__ void compute_RHS
    ```
    \[
    -\nabla \cdot \mathbf{F} =- \frac{\mathbf{F_x}_{i+\frac{1}{2},j,k} - \mathbf{F_x}_{i-\frac{1}{2},j,k}}{\Delta x}
                              - \frac{\mathbf{F_y}_{i,j+\frac{1}{2},k} - \mathbf{F_y}_{i,j-\frac{1}{2},k}}{\Delta y}
                              - \frac{\mathbf{F_z}_{i,j,k+\frac{1}{2}} - \mathbf{F_z}_{i,j,k-\frac{1}{2}}}{\Delta z}
    \]
    where
    \[
    \mathbf{F_x}_{i+\frac{1}{2},j,k} = \mathbf{F_x}^{+}_{i+\frac{1}{2},j,k} + \mathbf{F_x}^{-}_{i+\frac{1}{2},j,k},
    \]
    and $\mathbf{F}^{+}_{i+\frac{1}{2},j,k}$ represents the flux from left to right, $\mathbf{F}^{-}_{i+\frac{1}{2},j,k}$ represents the flux from right to left.


    ```C
    __global__ void reconstruct_interface_flux
    ```
    \[
    \begin{aligned}
    \mathbf{F}^{+}_{i+\frac{1}{2}} = \textbf{Minmod}(\mathbf{F}^{+}_{i-1}, \mathbf{F}^{+}_{i}, \mathbf{F}^{+}_{i+1}) \\
    \mathbf{F}^{-}_{i+\frac{1}{2}} = \textbf{Minmod}(\mathbf{F}^{-}_{i+2}, \mathbf{F}^{-}_{i+1}, \mathbf{F}^{-}_{i})
    \end{aligned}
    \]


    ```C
    __global__ void flux_vector_splitting
    ```
    \[
    \begin{aligned}
    \mathbf{F}^{+}_{i} = \mathbf{F}_i + \lambda_i \mathbf{U}_i \\
    \mathbf{F}^{-}_{i} = \mathbf{F}_i - \lambda_i \mathbf{U}_i
    \end{aligned}
    \]

    ```C
    __global__ void compute_primitive_vars_fluxes
    ```
    \[
    \begin{aligned}
    c_i = \sqrt{\gamma p_i/\rho_i} \\
    \lambda_{x,i} = |u_i| + c_i \\
    \lambda_{y,i} = |v_i| + c_i \\
    \lambda_{z,i} = |w_i| + c_i \\
    \lambda_{i} = \max(\lambda_{x,i}, \lambda_{y,i}, \lambda_{z,i})
    \end{aligned}
    \]
---


