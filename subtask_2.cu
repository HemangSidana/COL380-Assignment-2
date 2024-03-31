#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

__global__ void convolveKernel(float *inputTensor, int inputChannels, int inputSize,
                               float *outputTensor, int outputChannels, float *kernels, 
                               float *biases, int kernelSize, int outputSize) {
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    int outZ = blockIdx.z * blockDim.z + threadIdx.z;

    if(outX >= outputSize || outY >= outputSize || outZ >= outputChannels) return;

    float sum = 0.0f;
    for(int i = 0; i < inputChannels; i++) {
        for(int j = 0; j < kernelSize; j++) {
            for(int k = 0; k < kernelSize; k++) {
                int inX = outX + j;
                int inY = outY + k;
                sum += inputTensor[i*inputSize*inputSize + inX*inputSize + inY] *
                       kernels[outZ*inputChannels*kernelSize*kernelSize + i*kernelSize*kernelSize + j*kernelSize + k];
            }
        }
    }

    sum += biases[outZ];
    outputTensor[outZ*outputSize*outputSize + outX*outputSize + outY] = sum;
}

__global__ void optimizedConvolveKernel(float *inputTensor, int inputChannels, int inputSize,
                                        float *outputTensor, int outputChannels, float *kernels, 
                                        float *biases, int kernelSize, int outputSize) {
    extern __shared__ float s_kernels[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int channel = blockIdx.z;

    if (tx < kernelSize * kernelSize * inputChannels) {
        s_kernels[tx] = kernels[channel * kernelSize * kernelSize * inputChannels + tx];
    }
    __syncthreads();

    int outX = blockIdx.x * blockDim.x + tx;
    int outY = blockIdx.y * blockDim.y + ty;

    if(outX >= outputSize || outY >= outputSize || channel >= outputChannels) return;

    float sum = 0.0f;
    for(int i = 0; i < inputChannels; i++) {
        for(int j = 0; j < kernelSize; j++) {
            for(int k = 0; k < kernelSize; k++) {
                int inX = outX + j;
                int inY = outY + k;
                sum += inputTensor[i*inputSize*inputSize + inX*inputSize + inY] *
                       s_kernels[i*kernelSize*kernelSize + j*kernelSize + k];
            }
        }
    }

    sum += biases[channel];
    outputTensor[channel*outputSize*outputSize + outX*outputSize + outY] = sum;
}

__global__ void convolveWithPaddingKernel(float *inputTensor, int inputChannels, int inputSize,
                                          float *outputTensor, int outputChannels, float *kernels,
                                          float *biases, int kernelSize) {
    int padding = (kernelSize - 1) / 2;
    int outputSize = inputSize; // Assuming the output size is the same as the input size due to padding

    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    int outZ = blockIdx.z;

    if(outX < outputSize && outY < outputSize && outZ < outputChannels) {
        float sum = 0.0;
        for(int i = 0; i < inputChannels; i++) {
            for(int j = 0; j < kernelSize; j++) {
                for(int k = 0; k < kernelSize; k++) {
                    int inputRow = outX + j - padding;
                    int inputCol = outY + k - padding;
                    if(inputRow >= 0 && inputRow < inputSize && inputCol >= 0 && inputCol < inputSize) {
                        sum += inputTensor[i * inputSize * inputSize + inputRow * inputSize + inputCol] * 
                               kernels[outZ * inputChannels * kernelSize * kernelSize + i * kernelSize * kernelSize + j * kernelSize + k];
                    }
                }
            }
        }
        sum += biases[outZ];
        outputTensor[outZ * outputSize * outputSize + outX * outputSize + outY] = sum;
    }
}

__global__ void optimizedConvolveWithPaddingKernel(float *inputTensor, int inputChannels, int inputSize,
                                                   float *outputTensor, int outputChannels, float *kernels,
                                                   float *biases, int kernelSize) {
    extern __shared__ float sharedKernel[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int outZ = blockIdx.z;
    if (threadIdx.x < kernelSize * kernelSize * inputChannels) {
        sharedKernel[threadIdx.x] = kernels[outZ * kernelSize * kernelSize * inputChannels + threadIdx.x];
    }
    __syncthreads();

    int padding = (kernelSize - 1) / 2;
    int outX = blockIdx.x * blockDim.x + tx;
    int outY = blockIdx.y * blockDim.y + ty;
    int outputSize = inputSize; // With padding applied, output size matches the input size

    if(outX >= outputSize || outY >= outputSize || outZ >= outputChannels) return;

    float sum = 0.0;
    for(int i = 0; i < inputChannels; i++) {
        for(int j = 0; j < kernelSize; j++) {
            for(int k = 0; k < kernelSize; k++) {
                int inputRow = outX + j - padding;
                int inputCol = outY + k - padding;
                if(inputRow >= 0 && inputRow < inputSize && inputCol >= 0 && inputCol < inputSize) {
                    sum += inputTensor[i * inputSize * inputSize + inputRow * inputSize + inputCol] * 
                           sharedKernel[i * kernelSize * kernelSize + j * kernelSize + k];
                }
            }
        }
    }
    sum += biases[outZ];
    outputTensor[outZ * outputSize * outputSize + outX * outputSize + outY] = sum;
}


__global__ void ReLUKernel(float *inputTensor, float *outputTensor, int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        outputTensor[idx] = (inputTensor[idx] > 0.0f) ? inputTensor[idx] : 0.0f;
    }
}

__global__ void TanhKernel(float *inputTensor, float *outputTensor, int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        outputTensor[idx] = tanhf(inputTensor[idx]);
    }
}

__global__ void maxPoolingKernel(float *inputTensor, float *outputTensor, int inputChannels, int inputSize, int poolSize, int outputSize) {
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z;

    if (outX < outputSize && outY < outputSize && channel < inputChannels) {
        float maxVal = -FLT_MAX;
        for (int i = 0; i < poolSize; ++i) {
            for (int j = 0; j < poolSize; ++j) {
                int inX = outX * poolSize + i;
                int inY = outY * poolSize + j;
                float val = inputTensor[channel * inputSize * inputSize + inX * inputSize + inY];
                maxVal = fmaxf(maxVal, val);
            }
        }
        outputTensor[channel * outputSize * outputSize + outX * outputSize + outY] = maxVal;
    }
}

__global__ void avgPoolingKernel(float *inputTensor, float *outputTensor, int inputChannels, int inputSize, int poolSize, int outputSize) {
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z;

    if (outX < outputSize && outY < outputSize && channel < inputChannels) {
        float sum = 0.0f;
        for (int i = 0; i < poolSize; ++i) {
            for (int j = 0; j < poolSize; ++j) {
                int inX = outX * poolSize + i;
                int inY = outY * poolSize + j;
                sum += inputTensor[channel * inputSize * inputSize + inX * inputSize + inY];
            }
        }
        outputTensor[channel * outputSize * outputSize + outX * outputSize + outY] = sum / (poolSize * poolSize);
    }
}

__global__ void computeExponentialsKernel(float *inputVector, float *expValues, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        expValues[idx] = expf(inputVector[idx]);
    }
}

__global__ void normalizeKernel(float *expValues, float *probabilities, float expSum, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        probabilities[idx] = expValues[idx] / expSum;
    }
}

void softmaxCUDA(float *inputVector, int size, float *probabilities) {
    float *deviceInputVector, *deviceExpValues, *deviceProbabilities;
    float *expValues = new float[size]; // Temporarily store exponentials on host

    cudaMalloc(&deviceInputVector, size * sizeof(float));
    cudaMalloc(&deviceExpValues, size * sizeof(float));
    cudaMalloc(&deviceProbabilities, size * sizeof(float));

    cudaMemcpy(deviceInputVector, inputVector, size * sizeof(float), cudaMemcpyHostToDevice);

    // Assuming a simple block size, adjust as needed
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    // Compute exponentials
    computeExponentialsKernel<<<numBlocks, blockSize>>>(deviceInputVector, deviceExpValues, size);

    // Copy back exponentials to host to compute sum (Consider using parallel reduction on device for efficiency)
    cudaMemcpy(expValues, deviceExpValues, size * sizeof(float), cudaMemcpyDeviceToHost);
    float expSum = 0.0f;
    for (int i = 0; i < size; ++i) {
        expSum += expValues[i];
    }

    // Normalize
    normalizeKernel<<<numBlocks, blockSize>>>(deviceExpValues, deviceProbabilities, expSum, size);

    // Copy result back to host
    cudaMemcpy(probabilities, deviceProbabilities, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(deviceInputVector);
    cudaFree(deviceExpValues);
    cudaFree(deviceProbabilities);
    delete[] expValues;
}


int main(){

/*
    int sharedMemSize = kernelSize * kernelSize * inputChannels * sizeof(float);
    optimizedConvolveKernel<<<dimGrid, dimBlock, sharedMemSize>>>(...);

*/

    // dim3 threadsPerBlock(16, 16); // Example configuration, adjust based on your GPU's capabilities
    // dim3 numBlocks((outputSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
    //             (outputSize + threadsPerBlock.y - 1) / threadsPerBlock.y,
    //             outputChannels);
    // convolveWithPaddingKernel<<<numBlocks, threadsPerBlock>>>(inputTensor, inputChannels, inputSize, outputTensor, outputChannels, kernels, biases, kernelSize);




    return 0;
}