#include <iostream>
#include <cstdlib>
#include <cassert>
#include <ctime>
#include <chrono>
#include <random>
#include <limits>
#include <cmath>

#define ll long long
using namespace std;


float** create(int n){
    float** m= new float* [n];
    for(int i=0;i<n;i++){
        m[i]= new float[n];
    }
    return m;
}

void display(float** m, int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            cout<<m[i][j]<<" ";
        }
        cout<<'\n';
    }
}

void displayVector(const float* vector, int size) {
    for (int i = 0; i < size; ++i) {
        cout << vector[i] << " ";
    }
    cout <<endl;
}

float** convolution(float** input, int n, float** kernel, int k_size) {
    int outputSize = n - k_size + 1;

    float** output = create(outputSize);

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            output[i][j] = 0.0;
            for (int k = 0; k < k_size; ++k) {
                for (int l = 0; l < k_size; ++l) {
                    output[i][j] += input[i + k][j + l] * kernel[k][l];
                }
            }
        }
    }

    return output;
}

float** convolutionWithPadding(float** input, int n, float** kernel, int k_size) {
    int padding = k_size / 2; // Assuming k_size is odd 
    int paddedSize = n + 2 * padding;

    float** paddedInput = create(paddedSize);
    for(int i=0;i<paddedSize;i++){
        for(int j=0;j<paddedSize;j++){
            paddedInput[i][j]=0.0;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            paddedInput[i + padding][j + padding] = input[i][j];
        }
    }

    return convolution(paddedInput,paddedSize,kernel,k_size);
}

float** relu(float** input, int n) {
    float** output = create(n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            output[i][j] = max(0.0f, input[i][j]);
        }
    }
    return output;
}

float** tanh(float** input, int n) {
    float** output = create(n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            output[i][j] = tanh(input[i][j]);
        }
    }
    return output;
}

float** maxPooling(float** input,int n,int poolingSize) {
    int outputSize = n / poolingSize;
    float** output = create(outputSize);

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            float maxVal = input[i * poolingSize][j * poolingSize];
            for (int k = 0; k < poolingSize; ++k) {
                for (int l = 0; l < poolingSize; ++l) {
                    maxVal = max(maxVal, input[i * poolingSize + k][j * poolingSize + l]);
                }
            }
            output[i][j] = maxVal;
        }
    }

    return output;
}

float** avgPooling(float** input,int n,int poolingSize) {
    int outputSize = n / poolingSize;
    float** output = create(outputSize);

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            float avgVal = 0;
            for (int k = 0; k < poolingSize; ++k) {
                for (int l = 0; l < poolingSize; ++l) {
                    avgVal += input[i * poolingSize + k][j * poolingSize + l];
                }
            }
            output[i][j] = avgVal/(poolingSize*poolingSize);
        }
    }

    return output;
}

void softmax(float* input, int size) {
    float expSum = 0.0;

    for (int i = 0; i < size; ++i) {
        input[i] = exp(input[i]);
        expSum += input[i];
    }

    for (int i = 0; i < size; ++i) {
        input[i] /= expSum;
    }
}

void sigmoid(float* input, int size) {
    for (int i = 0; i < size; ++i) {
        input[i] = 1.0 / (1.0 + exp(-input[i]));
    }
}


int main(int argc, const char * argv[]){
    int n= stoi(argv[1]);
    int k_size= stoi(argv[2]);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    cout<<"hello"<<endl;

    
    float**input= create(n);
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            input[i][j]= dist(gen);
        }
    }

    float**kernal= create(k_size);
    for(int i=0;i<k_size;i++){
        for(int j=0;j<k_size;j++){
            kernal[i][j]= dist(gen);
        }
    }

    float** conv = convolution(input,n,kernal,k_size);
    float** conv_pad = convolutionWithPadding(input,n,kernal,k_size);
    float** relu_matrix = relu(input,n);
    float** tanh_matrix = tanh(input,n);
    float** max_pool = maxPooling(input,n,2);
    float** avg_pool = avgPooling(input,n,2);

    float* soft = new float[n];
    float* sig = new float[n];
    for(int j=0;j<k_size;j++){
        soft[j] = dist(gen);
        sig[j] = dist(gen);
    }
    softmax(soft,n);
    sigmoid(sig,n);

    display(conv,n-k_size+1);
    cout<<endl;
    display(conv_pad,n);
    cout<<endl;
    display(relu_matrix,n);
    cout<<endl;
    display(tanh_matrix,n);
    cout<<endl;
    display(max_pool,n/2);
    cout<<endl;
    display(avg_pool,n/2);
    cout<<endl;
    displayVector(soft,n);
    cout<<endl;
    displayVector(sig,n);

    return 0;
}