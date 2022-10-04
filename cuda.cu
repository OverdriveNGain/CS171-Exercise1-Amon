#include <stdio.h>
#include <cuda.h>

// Kernel
__global__ void increment_vector(float *matrix1, int dimensionLength)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < dimensionLength*dimensionLength) {
        matrix1[i] += 1;
        // printf("%f\n", matrix1[i]);
    }
}

__global__
void kernel_1t1e(){
    
}

__global__
void kernel_1t1r(){
    
}

__global__
void kernel_1t1c(){
    
}

__host__
void matrixAdd(float** output, float** matrix1, float** matrix2, int dimensionLength){
    size_t arrayByteSize = dimensionLength*dimensionLength*sizeof(float);

    // Allocate memory for arrays d_A, d_B, and d_C on device
    float *matrix1_d, *matrix2_d, *matrixOutput_d;
    cudaMalloc(&matrix1_d, arrayByteSize);
    cudaMalloc(&matrix2_d, arrayByteSize);
    cudaMalloc(&matrixOutput_d, arrayByteSize);

    // Copy data from host arrays A and B to device arrays d_A and d_B
    // cudaMemcpy(matrix1_d, *matrix1, floatArrSize, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix1_d, *matrix1, arrayByteSize, cudaMemcpyHostToDevice);
    // cudaMemcpy(matrix2_d, *matrix2, floatArrSize, cudaMemcpyHostToDevice);

    // Launch kernel
    increment_vector<<< 1, dimensionLength * dimensionLength >>>(matrix1_d, dimensionLength);

    // Copy data from device array d_C to host array C
    cudaMemcpy(*matrix1, matrix1_d, arrayByteSize, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(matrix1_d);
    cudaFree(matrix2_d);
    cudaFree(matrixOutput_d);
}

__host__
void matrixAlloc(float** matrix, int dimensionLength){
    *matrix = (float*) malloc(dimensionLength*dimensionLength*sizeof(float));
}

__host__
void matrixPrint(float** matrix, int dimensionLength){
    for (int i = 0; i < dimensionLength; i++){
        if (i == 0)
            printf("{{ ");
        else
            printf(" { ");
        
        for (int j = 0; j < dimensionLength; j++){
            if (j == dimensionLength - 1)
                printf("%f ", (*matrix)[i * dimensionLength + j]);
            else
                printf("%f, ", (*matrix)[i * dimensionLength + j]);
        }

        if (i == dimensionLength - 1)
            printf("}}\n");
        else
            printf("}\n");
    }
}

__host__
void matrixInitRandomValues(float** matrix, int dimensionLength, float maxValue){
    for (int i = 0; i < dimensionLength * dimensionLength; i++){
        // https://stackoverflow.com/questions/13408990/how-to-generate-random-float-number-in-c
        (*matrix)[i] = ((float)rand()/(float)(RAND_MAX)) * maxValue;
    }
}

// Main program
int main()
{
    int dimensionLength = 3;
    size_t arrayByteSize = dimensionLength*dimensionLength*sizeof(float);
    float *matrix1;
    float *matrix2;
    float *matrixOutput;

    matrix1 = (float*) malloc(arrayByteSize);
    matrix2 = (float*) malloc(arrayByteSize);
    matrixOutput = (float*) malloc(arrayByteSize);

    matrixInitRandomValues(&matrix1, dimensionLength, 100);
    matrixInitRandomValues(&matrix2, dimensionLength, 100);

    matrixPrint(&matrix1, dimensionLength);

    matrixAdd(&matrixOutput, &matrix1, &matrix2, dimensionLength);

    // Verify results
    matrixPrint(&matrix1, dimensionLength);

    // Free CPU memory
    free(matrix1);
    free(matrix2);
    free(matrixOutput);

    printf("SUCCESS!\n");
    return 0;
}