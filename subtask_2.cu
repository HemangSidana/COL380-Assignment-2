#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;


__global__ void convolutionKernel(float *inputMatrix, int inputSize, float *kernelMatrix, int kernelSize,
    float *outputMatrix, int paddingValue) {
    int outputSize = inputSize - kernelSize + 1 + 2 * paddingValue; // Output size considering padding

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < outputSize && col < outputSize) {
        float sum = 0.0f;
        for (int ki = 0; ki < kernelSize; ki++) {
            for (int kj = 0; kj < kernelSize; kj++) {
                int inputRowIndex = row + ki - paddingValue;
                int inputColIndex = col + kj - paddingValue;
                if (inputRowIndex >= 0 && inputRowIndex < inputSize && inputColIndex >= 0 && inputColIndex < inputSize) {
                    sum += inputMatrix[inputRowIndex * inputSize + inputColIndex] * kernelMatrix[ki * kernelSize + kj];
                }
            }
        }
        // Store the result in the output matrix
        outputMatrix[row * outputSize + col] = sum;
    }
}


__global__ void reluKernel(float *inputMatrix, int size, float *outputMatrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        outputMatrix[idx] = (inputMatrix[idx] > 0.0f) ? inputMatrix[idx] : 0.0f;
    }
}

__global__ void tanhKernel(float *inputMatrix, int size, float *outputMatrix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        outputMatrix[idx] = tanh(inputMatrix[idx]);
    }
}


__global__ void maxpoolKernel(float *inputMatrix, int inputSize, int poolSize, float *outputMatrix) {
    int outputSize = inputSize / poolSize;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < outputSize && col < outputSize) {
        float maxVal = inputMatrix[row * poolSize * inputSize + col * poolSize];
        for (int k = 0; k < poolSize; k++) {
            for (int l = 0; l < poolSize; l++) {
                float currentVal = inputMatrix[(row * poolSize + k) * inputSize + col * poolSize + l];
                if (currentVal > maxVal) {
                    maxVal = currentVal;
                }
            }
        }
        outputMatrix[row * outputSize + col] = maxVal;
    }
}

__global__ void avgpoolKernel(float *inputMatrix, int inputSize, int poolSize, float *outputMatrix) {
    int outputSize = inputSize / poolSize;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < outputSize && col < outputSize) {
        float sum = 0.0f;
        for (int k = 0; k < poolSize; k++) {
            for (int l = 0; l < poolSize; l++) {
                sum += inputMatrix[(row * poolSize + k) * inputSize + col * poolSize + l];
            }
        }
        outputMatrix[row * outputSize + col] = sum / (poolSize * poolSize);
    }
}


__global__ void softmax(float* inputVector, int inputsize ,float* outputVector) {
    float* expValues = new float[inputsize];
    float expSum = 0.0f;

    for (int i = 0; i < inputsize; ++i) {
        expValues[i] = expf(inputVector[i]);
        expSum += expValues[i];
    }

    for (int i = 0; i < inputsize; ++i) {
        outputVector[i] = expValues[i] / expSum;
    }

}


__global__ void sigmoid(float* inputVector, int inputSize, float* outputVector) {
    for (int i = 0; i < inputSize; ++i) {
        outputVector[i] = 1 / (1 + expf(-inputVector[i]));
    }
}


// Main function
int main(int argc, char *argv[]) {
    int task = atoi(argv[1]);

    switch (task) {
        case 1: {
            // Task 1: Convolution
            int inputSize = atoi(argv[2]);
            int kernelSize = atoi(argv[3]);
            int paddingValue = atoi(argv[4]);
            // Extract input values
            float* inputmatrix = new float[inputSize*inputSize];
            float* kernalmatrix = new float[kernelSize*kernelSize];
            int outputSize = inputSize - kernelSize + 1 + 2 * paddingValue;
            float* outputmatrix = new float[outputSize*outputSize];
            for (int i = 0; i < inputSize*inputSize; i++) {
                inputmatrix[i] = atof(argv[i+5]);
            }
            for (int i = 0; i < kernelSize*kernelSize; i++) {
                kernalmatrix[i] = atof(argv[i+5+inputSize*inputSize]);
            }

            float *device_inputmatrix, *device_kernalmatrix, *device_outputmatrix;
            cudaMalloc((void**)&device_inputmatrix, inputSize*inputSize*sizeof(float));
            cudaMalloc((void**)&device_kernalmatrix, kernelSize*kernelSize*sizeof(float));
            cudaMalloc((void**)&device_outputmatrix, outputSize*outputSize*sizeof(float));

            cudaMemcpy(device_inputmatrix, inputmatrix, inputSize*inputSize*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(device_kernalmatrix, kernalmatrix, kernelSize*kernelSize*sizeof(float), cudaMemcpyHostToDevice);


            // Launch convolution kernel
            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((outputSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (outputSize + threadsPerBlock.y - 1) / threadsPerBlock.y);
            convolutionKernel<<<numBlocks, threadsPerBlock>>>(device_inputmatrix, inputSize, device_kernalmatrix, kernelSize,
                                                               device_outputmatrix, paddingValue);

            // Free allocated memory
            cudaFree(device_inputmatrix);
            cudaFree(device_kernalmatrix);

            break;
        }
        case 2: {
            int type = atoi(argv[2]);
            int N = atoi(argv[3]);
            int M = atoi(argv[4]);
            int size = N * M;

            // Extract input values
            float* inputmatrix = new float[N*M];
            float* outputmatrix = new float[N*M];
            for (int i = 0; i < N*M; i++) {
                inputmatrix[i] = atof(argv[i+5]);
            }

            float *device_inputmatrix, *device_outputmatrix;
            cudaMalloc((void**)&device_inputmatrix, size*sizeof(float));
            cudaMalloc((void**)&device_outputmatrix, size*sizeof(float));

            cudaMemcpy(device_inputmatrix, inputmatrix, size*sizeof(float), cudaMemcpyHostToDevice);


            // Launch kernel based on type
            if (type == 0) {
                reluKernel<<<(size + 255) / 256, 256>>>(device_inputmatrix, size, device_outputmatrix);
            } else {
                tanhKernel<<<(size + 255) / 256, 256>>>(device_inputmatrix, size, device_outputmatrix);
            }


            // Free allocated memory
            cudaFree(device_inputmatrix);
            cudaFree(device_outputmatrix);

            break;
        }
        case 3: {
            int type = atoi(argv[2]);
            int inputSize = atoi(argv[3]);
            int poolSize = atoi(argv[4]);
            int outputSize = inputSize / poolSize;

            // Extract input values
            float* inputmatrix = new float[inputSize*inputSize];
            float* outputmatrix = new float[outputSize*outputSize];
            for (int i = 0; i < inputSize*inputSize; i++) {
                inputmatrix[i] = atof(argv[i+5]);
            }

            float *device_inputmatrix, *device_outputmatrix;
            cudaMalloc((void**)&device_inputmatrix, inputSize*inputSize*sizeof(float));
            cudaMalloc((void**)&device_outputmatrix, outputSize*outputSize*sizeof(float));

            cudaMemcpy(device_inputmatrix, inputmatrix, inputSize*inputSize*sizeof(float), cudaMemcpyHostToDevice);

            if (type == 0) {
                maxpoolKernel<<<(outputSize + 255) / 256, 256>>>(device_inputmatrix, inputSize, poolSize, device_outputmatrix);
            } else {
                avgpoolKernel<<<(outputSize + 255) / 256, 256>>>(device_inputmatrix, inputSize, poolSize, device_outputmatrix);
            }

            cudaDeviceSynchronize();

            cudaFree(inputMatrix);
            cudaFree(outputMatrix);

            break;
        }
        case 4: {
            int type = atoi(argv[2]);

            int inputSize = argc - 3;
            float *inputVector = new float[inputSize];
            float *outputVector = new float[inputSize];

            for (int i = 0; i < inputSize; ++i) {
                inputVector[i] = atof(argv[i + 3]);
            }

            float *device_inputvector, *device_outputvector;
            cudaMalloc((void**)&device_inputvector, inputSize*sizeof(float));
            cudaMalloc((void**)&device_outputvector, inputSize*sizeof(float));

            cudaMemcpy(device_inputvector, inputVector, inputSize*sizeof(float), cudaMemcpyHostToDevice);

            if (type == 0) {
                sigmoidKernel<<<(inputSize + 255) / 256, 256>>>(device_inputvector, inputSize, device_outputvector);
            } else {
                softmaxKernel<<<(inputSize + 255) / 256, 256>>>(device_inputvector, inputSize, device_outputvector);
            }

            cudaDeviceSynchronize();

            // Free allocated memory
            cudaFree(inputVector);
            cudaFree(outputVector);

            break;
        }
        default:
            cout << "Invalid task number!" << endl;
            return 1;
    }

    return 0;
}
