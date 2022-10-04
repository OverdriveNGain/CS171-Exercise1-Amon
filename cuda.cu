#include <stdio.h>
#include <cuda.h>

__global__
void kernel_1t1e(float *matrixOut, float *matrix1, float *matrix2, int matrixDimLen){
    int length1D = matrixDimLen * matrixDimLen;
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < length1D) matrixOut[i]  = matrix1[i] + matrix2[i];
}

__global__
void kernel_1t1r(float *matrixOut, float *matrix1, float *matrix2, int matrixDimLen){
    int length1D = matrixDimLen * matrixDimLen;
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    for (int j = 0; j < matrixDimLen; j++)
        if (i + j < length1D) matrixOut[i + j]  = matrix1[i + j] + matrix2[i + j];
}

__global__
void kernel_1t1c(float *matrixOut, float *matrix1, float *matrix2, int matrixDimLen){
    int length1D = matrixDimLen * matrixDimLen;
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    for (int j = 0; j < matrixDimLen; j++){
        int index = i + j * matrixDimLen;
        if (index < length1D) matrixOut[index]  = matrix1[index] + matrix2[index];
    }
}

__host__
void matrixAdd(float*** output, float*** matrix1, float*** matrix2, int dimensionLength){
    int flattenedLength = dimensionLength*dimensionLength;
    size_t arrayByteSize = flattenedLength*sizeof(float);

    // Flatten input arrays
    float* matrix1Flat = (float*) malloc(arrayByteSize);
    float* matrix2Flat = (float*) malloc(arrayByteSize);
    float* outputFlat = (float*) malloc(arrayByteSize);

    // Copy data to flattened input arrays
    for (int i = 0; i < dimensionLength; i++){
        for (int j = 0; j < dimensionLength; j++){
            matrix1Flat[i * dimensionLength + j] = (*matrix1)[i][j];
            matrix2Flat[i * dimensionLength + j] = (*matrix2)[i][j];
            outputFlat[i * dimensionLength + j] = (*output)[i][j];
        }
    }

    // Allocate memory for arrays on device
    float *matrix1_d, *matrix2_d, *matrixOutput_d;
    cudaMalloc(&matrix1_d, arrayByteSize);
    cudaMalloc(&matrix2_d, arrayByteSize);
    cudaMalloc(&matrixOutput_d, arrayByteSize);

    // Copy to device
    cudaMemcpy(matrix1_d, matrix1Flat, arrayByteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix2_d, matrix2Flat, arrayByteSize, cudaMemcpyHostToDevice);

    // Launch kernel
    // make more reasonable
    int threadBlockCount = ceil(flattenedLength/1024.0);
    int threadCountPerBlock = 1024;
    kernel_1t1e<<< threadBlockCount, threadCountPerBlock >>>(matrixOutput_d, matrix1_d, matrix2_d, dimensionLength);
    // kernel_1t1r<<< threadBlockCount, threadCountPerBlock >>>(matrixOutput_d, matrix1_d, matrix2_d, dimensionLength);
    // kernel_1t1c<<< threadBlockCount, threadCountPerBlock >>>(matrixOutput_d, matrix1_d, matrix2_d, dimensionLength);

    // Copy data from device output array to flattened host array
    cudaMemcpy(outputFlat, matrixOutput_d, arrayByteSize, cudaMemcpyDeviceToHost);

    // Copy data from flattened output array to output array
    for (int i = 0; i < dimensionLength; i++){
        for (int j = 0; j < dimensionLength; j++){
            (*output)[i][j] = outputFlat[i * dimensionLength + j];
        }
    }

    // Free GPU memory
    cudaFree(matrix1_d);
    cudaFree(matrix2_d);
    cudaFree(matrixOutput_d);

    // Free flattened input arrays
    free(matrix1Flat);
    free(matrix2Flat);
    free(outputFlat);
}

__host__
void matrixAlloc(float** matrix, int dimensionLength){
    *matrix = (float*) malloc(dimensionLength*dimensionLength*sizeof(float));
}

__host__
void matrixPrint(float*** matrix, int dimensionLength){
    for (int i = 0; i < dimensionLength; i++){
        if (i == 0)
            printf("{{ ");
        else
            printf(" { ");
        
        for (int j = 0; j < dimensionLength; j++){
            if (j == dimensionLength - 1)
                printf("%f ", (*matrix)[i][j]);
            else
                printf("%f, ", (*matrix)[i][j]);
        }

        if (i == dimensionLength - 1)
            printf("}}\n");
        else
            printf("}\n");
    }
}

__host__
void matrixInitRandomValues(float*** matrix, int dimensionLength, float maxValue){
    for (int i = 0; i < dimensionLength; i++){
        for (int j = 0; j < dimensionLength; j++){
            // https://stackoverflow.com/questions/13408990/how-to-generate-random-float-number-in-c
            (*matrix)[i][j] = ((float)rand()/(float)(RAND_MAX)) * maxValue;
        }
    }
}

// Main program
int main()
{
    // Initialization of variables
    int dimensionLength = 3;
    size_t arrayByteSizeP = dimensionLength*sizeof(float*);
    size_t arrayByteSizeF = dimensionLength*sizeof(float);
    float **matrix1;
    float **matrix2;
    float **matrixOutput;

    // Memory allocation of matrices
    matrix1 = (float**) malloc(arrayByteSizeP);
    matrix2 = (float**) malloc(arrayByteSizeP);
    matrixOutput = (float**) malloc(arrayByteSizeP);
    for (int i = 0; i < dimensionLength; i++){
        matrix1[i] = (float*) malloc(arrayByteSizeF);
        matrix2[i] = (float*) malloc(arrayByteSizeF);
        matrixOutput[i] = (float*) malloc(arrayByteSizeF);
    }

    // Assignment of random values to matrix 1 and 2
    matrixInitRandomValues(&matrix1, dimensionLength, 100);
    matrixInitRandomValues(&matrix2, dimensionLength, 100);

    // Initial printing of values of matrix 1 and 2
    matrixPrint(&matrix1, dimensionLength);
    matrixPrint(&matrix2, dimensionLength);

    // Adding of matrices
    matrixAdd(&matrixOutput, &matrix1, &matrix2, dimensionLength);

    // Printing of output matrix
    matrixPrint(&matrixOutput, dimensionLength);

    // Free CPU memory
    for (int i = 0; i < dimensionLength; i++){
        free(matrix1[i]);
        free(matrix2[i]);
        free(matrixOutput[i]);
    }
    free(matrix1);
    free(matrix2);
    free(matrixOutput);

    printf("Success!\n");
    return 0;
}