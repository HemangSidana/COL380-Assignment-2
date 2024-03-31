#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <fstream>
using namespace std;

void convolve_without_padding(float *inputTensor, int inputChannels, int inputSize, float *outputTensor, int outputChannels, float *kernels, float *biases, int kernelSize)
{
    int outputSize = inputSize - kernelSize + 1;

    for(int a=0;a<outputChannels;a++){
        for(int b=0;b<outputSize;b++){
            for(int c=0;c<outputSize;c++){
                float sum=0.0;
                for(int i=0;i<inputChannels;i++){
                    for(int j=0;j<kernelSize;j++){
                        for(int k=0;k<kernelSize;k++){
                            sum+= inputTensor[i*inputSize*inputSize + (b+j)*inputSize + (c+k)] * kernels[a*inputChannels*kernelSize*kernelSize + i*kernelSize*kernelSize + j*kernelSize +k];
                        }
                    }
                }
                sum+= biases[a];
                outputTensor[a*outputSize*outputSize + b*outputSize + c] = sum;
            }
        }
    }
}

void convolve_with_padding(float *inputTensor, int inputChannels, int inputSize, float *outputTensor, int outputChannels, float *kernels, float *biases, int kernelSize) {
    int padding = (kernelSize - 1) / 2; // Compute padding size
    int paddedSize = inputSize + 2 * padding; // Compute padded input size

    for(int a = 0; a < outputChannels; a++) {
        for(int b = 0; b < inputSize; b++) {
            for(int c = 0; c < inputSize; c++) {
                float sum = 0.0;
                for(int i = 0; i < inputChannels; i++) {
                    for(int j = 0; j < kernelSize; j++) {
                        for(int k = 0; k < kernelSize; k++) {
                            int inputRow = b + j - padding;
                            int inputCol = c + k - padding;
                            if(inputRow >= 0 && inputRow < inputSize && inputCol >= 0 && inputCol < inputSize) {
                                sum += inputTensor[i * inputSize * inputSize + inputRow * inputSize + inputCol] * kernels[a * inputChannels * kernelSize * kernelSize + i * kernelSize * kernelSize + j * kernelSize + k];
                            }
                        }
                    }
                }
                sum += biases[a];
                outputTensor[a * inputSize * inputSize + b * inputSize + c] = sum;
            }
        }
    }
}

void Relu(float *inputTensor, float *outputTensor, int inputChannels, int inputSize) {
    for (int inChannel = 0; inChannel < inputChannels; inChannel++) {
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                int index = inChannel * inputSize * inputSize + i * inputSize + j;
                outputTensor[index] = (inputTensor[index] > 0.0f) ? inputTensor[index] : 0.0f;
            }
        }
    }
}

void Tanh(float *inputTensor, float *outputTensor, int inputChannels, int inputSize) {
    for (int inChannel = 0; inChannel < inputChannels; inChannel++) {
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                int index = inChannel * inputSize * inputSize + i * inputSize + j;
                outputTensor[index] = tanh(inputTensor[index]);
            }
        }
    }
}

void maxpooling(float *inputTensor, float *outputTensor, int inputChannels, int inputSize, int poolSize) {
    int outputSize = inputSize / poolSize;

    for (int inChannel = 0; inChannel < inputChannels; inChannel++) {
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                float maxval = inputTensor[inChannel * inputSize * inputSize + i * poolSize * inputSize + j * poolSize];
                for (int k = 0; k < poolSize; k++) {
                    for (int l = 0; l < poolSize; l++) {
                        float val = inputTensor[inChannel * inputSize * inputSize + (i * poolSize + k) * inputSize + j * poolSize + l];
                        maxval = (val > maxval) ? val : maxval;
                    }
                }
                outputTensor[inChannel * outputSize * outputSize + i * outputSize + j] = maxval;
            }
        }
    }
}


void avgpooling(float *inputTensor, float *outputTensor, int inputChannels, int inputSize, int poolSize) {
    int outputSize = inputSize / poolSize;

    for (int inChannel = 0; inChannel < inputChannels; inChannel++) {
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                float sum = 0.0f;
                for (int k = 0; k < poolSize; k++) {
                    for (int l = 0; l < poolSize; l++) {
                        float val = inputTensor[inChannel * inputSize * inputSize + (i * poolSize + k) * inputSize + j * poolSize + l];
                        sum += val ;
                    }
                }
                outputTensor[inChannel * outputSize * outputSize + i * outputSize + j] = sum / (poolSize * poolSize);
            }
        }
    }
}


void softmax(float* inputVector, int size ,float* probabilities) {
    float* expValues = new float[size];
    float expSum = 0.0f;

    for (int i = 0; i < size; ++i) {
        expValues[i] = expf(inputVector[i]);
        expSum += expValues[i];
    }

    for (int i = 0; i < size; ++i) {
        probabilities[i] = expValues[i] / expSum;
    }

}


int main(){

    return 0;
}
