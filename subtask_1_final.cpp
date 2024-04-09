#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <fstream>
using namespace std;


void convolution(float *inputMatrix, int inputSize, float *kernelMatrix, int kernelSize, float *outputMatrix, int paddingValue) {
    int outputSize = inputSize - kernelSize + 1 + 2 * paddingValue; // Output size considering padding

    // Loop over each element in the output matrix
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            float sum = 0.0f;
            // Loop over each element in the kernel
            for (int ki = 0; ki < kernelSize; ki++) {
                for (int kj = 0; kj < kernelSize; kj++) {
                    // Compute input matrix indices considering padding
                    int inputRowIndex = i + ki - paddingValue;
                    int inputColIndex = j + kj - paddingValue;
                    // Check for valid input matrix indices
                    if (inputRowIndex >= 0 && inputRowIndex < inputSize && inputColIndex >= 0 && inputColIndex < inputSize) {
                        // Perform element-wise multiplication and accumulation
                        sum += inputMatrix[inputRowIndex * inputSize + inputColIndex] * kernelMatrix[ki * kernelSize + kj];
                    }
                }
            }
            // Store the result in the output matrix
            outputMatrix[i * outputSize + j] = sum;
        }
    }
}


void relu(float *inputMatrix, int m, int n, float *outputMatrix) {
    // Apply ReLU function element-wise
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int index = i * n + j;
            outputMatrix[index] = (inputMatrix[index] > 0.0f) ? inputMatrix[index] : 0.0f;
        }
    }
}

void tanh(float *inputMatrix, int m, int n, float *outputMatrix) {
    // Apply Tanh function element-wise
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int index = i * n + j;
            outputMatrix[index] = tanh(inputMatrix[index]);
        }
    }
}


void maxpool(float *inputMatrix, int inputSize, int poolSize, float *outputMatrix) {
    int outputSize = inputSize / poolSize;

    // Loop through each pooling window
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            // Find maximum value within the pooling window
            float maxVal = inputMatrix[i * poolSize * inputSize + j * poolSize];
            for (int k = 0; k < poolSize; k++) {
                for (int l = 0; l < poolSize; l++) {
                    float currentVal = inputMatrix[(i * poolSize + k) * inputSize + j * poolSize + l];
                    if (currentVal > maxVal) {
                        maxVal = currentVal;
                    }
                }
            }
            // Store the maximum value in the output matrix
            outputMatrix[i * outputSize + j] = maxVal;
        }
    }
}



void avgpool(float *inputMatrix, int inputSize, int poolSize, float *outputMatrix) {
    int outputSize = inputSize / poolSize;

    // Loop through each pooling window
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            // Calculate the average value within the pooling window
            float sum = 0.0f;
            for (int k = 0; k < poolSize; k++) {
                for (int l = 0; l < poolSize; l++) {
                    sum += inputMatrix[(i * poolSize + k) * inputSize + j * poolSize + l];
                }
            }
            float avgVal = sum / (poolSize * poolSize);
            // Store the average value in the output matrix
            outputMatrix[i * outputSize + j] = avgVal;
        }
    }
}



void softmax(float* inputVector, int inputsize ,float* outputVector) {
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


void sigmoid(float* inputVector, int inputSize, float* outputVector) {
    for (int i = 0; i < inputSize; ++i) {
        outputVector[i] = 1 / (1 + expf(-inputVector[i]));
    }
}


int main(int argc, char *argv[]) {
    
    cout<<endl;
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
            for (int i = 0; i < inputSize*inputSize; i++) {
                inputmatrix[i] = atof(argv[i+5]);
            }
            for (int i = 0; i < kernelSize*kernelSize; i++) {
                kernalmatrix[i] = atof(argv[i+5+inputSize*inputSize]);
            }

            // for (int i = 0; i < inputSize; ++i) {
            //     for (int j = 0; j < inputSize; ++j) {
            //         cout << inputmatrix[i * inputSize + j] << " ";
            //     }cout << endl;
            // }cout << endl;

            // for (int i = 0; i < kernelSize; ++i) {
            //     for (int j = 0; j < kernelSize; ++j) {
            //         cout << kernalmatrix[i * kernelSize + j] << " ";
            //     }cout << endl;
            // }cout << endl;



            int outputSize = inputSize - kernelSize + 1 + 2 * paddingValue;
            float* outputmatrix = new float[outputSize*outputSize];

            convolution(inputmatrix,inputSize,kernalmatrix,kernelSize,outputmatrix,paddingValue);

            // for (int i = 0; i < outputSize; ++i) {
            //     for (int j = 0; j < outputSize; ++j) {
            //         cout << outputmatrix[i * outputSize + j] << " ";
            //     }cout << endl;
            // }

            break;
        }
        case 2: {

            int type = atoi(argv[2]);

            int N = atoi(argv[3]);
            int M = atoi(argv[4]);
            
            // Extract input values
            float* inputmatrix = new float[N*M];
            for (int i = 0; i < N*M; i++) {
                inputmatrix[i] = atof(argv[i+5]);
            }

    
            float* outputmatrix = new float[N*M];
            if(type==0){
                relu(inputmatrix,N,M,outputmatrix);
            }
            else{
                tanh(inputmatrix,N,M,outputmatrix);
            }

            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < M; ++j) {
                    cout << outputmatrix[i * M + j] << " ";
                }cout << endl;
            }
            
            break;
        }
        case 3: {
            int type = atoi(argv[2]);
            int poolSize = atoi(argv[3]);
            int inputSize = atoi(argv[4]);
            // Extract input values
            float* inputmatrix = new float[inputSize*inputSize];
            for (int i = 0; i < inputSize*inputSize; i++) {
                inputmatrix[i] = atof(argv[i+5]);
            }
            
            int outputSize = inputSize/poolSize;
            float* outputmatrix = new float[outputSize*outputSize];

            for (int i = 0; i < inputSize; ++i) {
                for (int j = 0; j < inputSize; ++j) {
                    cout << inputmatrix[i * inputSize + j] << " ";
                }cout << endl;
            }cout << endl;

            if(type==0){
                maxpool(inputmatrix,inputSize,poolSize,outputmatrix);
            }
            else{
                avgpool(inputmatrix,inputSize,poolSize,outputmatrix);
            }

            for (int i = 0; i < outputSize; ++i) {
                for (int j = 0; j < outputSize; ++j) {
                    cout << outputmatrix[i * outputSize + j] << " ";
                }cout << endl;
            }

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

            if(type==0){
                sigmoid(inputVector, inputSize, outputVector);
            }
            else{
                softmax(inputVector, inputSize, outputVector);
            }

            for (int i = 0; i < inputSize; ++i) {
                cout << outputVector[i] << " ";   
            }cout << endl;

            break;
        }
        default:
            cout << "Invalid task number!" << endl;
            return 1;
    }

    return 0;
}
