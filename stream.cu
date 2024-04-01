#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
using namespace std;

#define num_streams 1

__global__ void Conv_1(float* input, float* output, float* kernel, float* bias) {
    int channel = blockIdx.x;
    int a = threadIdx.x; 
    int b = threadIdx.y;
    float sum = 0.0;
    int input_size = 28;
    int kernel_size = 5;
    int output_size = 24;
    
    for(int i = 0; i < kernel_size; i++) {
        for(int j = 0; j < kernel_size; j++) {
            int x = i + a; 
            int y = j + b;
            // sum += kernel[channel][i][j]* input[x][y]
            sum += kernel[(channel * kernel_size * kernel_size) + (i * kernel_size) + j] * input[(x * input_size) + y];
        }
    }
    sum += bias[channel];
    // output[channel][a][b] = sum
    output[(channel * output_size * output_size) + (a * output_size) + b] = sum;
}

__global__ void Pool_1(float* input, float* output) {
    int channel = blockIdx.x;
    int a = threadIdx.x;
    int b = threadIdx.y;
    int input_size = 24;
    int output_size = 12;
    int input_channel_size = input_size * input_size;
    int output_channel_size = output_size * output_size;

    float max_val = input[(channel * input_channel_size) + ((2 * a) * input_size) + (2 * b)];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            // val= input[channel][2*a+i][2*b+j]
            float val = input[(channel * input_channel_size) + ((2 * a + i) * input_size) + (2 * b + j)];
            max_val = fmaxf(max_val, val);
        }
    }
    // output[channel][a][b]= max_val
    output[(channel * output_channel_size) + (a * output_size) + b] = max_val;
}

__global__ void Conv_2(float* input, float* output, float* kernel, float* bias) {
    int chx = blockIdx.x;
    int thx = threadIdx.x;
    int thy = threadIdx.y;
    int input_size = 12;
    int output_size = 8;
    int kernel_size = 5;
    int input_channel_size = input_size * input_size;
    int output_channel_size = output_size * output_size;
    int kernel_channel_size = kernel_size * kernel_size;

    float sum = 0.0;
    for (int c = 0; c < 20; ++c) {
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                int x = i + thx;
                int y = j + thy;
                float input_val = input[(c * input_channel_size) + (x * input_size) + y];
                float kernel_val = kernel[(chx * 20 * kernel_channel_size) + (c * kernel_channel_size) + (i * kernel_size) + j];
                sum += kernel_val * input_val;
            }
        }
    }
    sum += bias[chx];
    // output[chx][thx][thy]= sum
    output[(chx * output_channel_size) + (thx * output_size) + thy] = sum;
}


__global__ void Pool_2(float* input, float* output) {
    int channel = blockIdx.x;
    int a = threadIdx.x;
    int b = threadIdx.y;
    int input_size = 8;
    int output_size = 4;
    int input_channel_size = input_size * input_size;
    int output_channel_size = output_size * output_size;

    float max_val = input[(channel * input_channel_size) + (2 * a * input_size) + (2 * b)];
    max_val = fmaxf(max_val, input[(channel * input_channel_size) + (2 * a + 1) * input_size + (2 * b)]);
    max_val = fmaxf(max_val, input[(channel * input_channel_size) + (2 * a) * input_size + (2 * b + 1)]);
    max_val = fmaxf(max_val, input[(channel * input_channel_size) + (2 * a + 1) * input_size + (2 * b + 1)]);

    // output[channel][a][b]= max_val
    output[(channel * output_channel_size) + (a * output_size) + b] = max_val;
}


__global__ void FC_1(float* input, float* output, float* kernel, float* bias) {
    int idx = threadIdx.x;
    float sum = 0.0;
    int input_size = 4;
    int kernel_channels = 50;
    for (int c = 0; c < kernel_channels; ++c) {
        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                int input_index = (c * input_size * input_size) + (i * input_size) + j;
                int kernel_index = (idx * kernel_channels * input_size * input_size) + (c * input_size * input_size) + (i * input_size) + j;
                // sum += kernel[idx][c][i][j]* input[c][i][j]
                sum += kernel[kernel_index] * input[input_index];
            }
        }
    }
    sum += bias[idx];
    output[idx] = fmaxf(0.0, sum);
}


__global__ void FC_2(float* input, float* output, float* kernel, float* bias) {
    int idx = threadIdx.x;
    float sum = 0.0;
    int input_size = 500;
    for (int i = 0; i < input_size; ++i) {
        int kernel_index = (idx * input_size) + i;
        sum += kernel[kernel_index] * input[i];
    }
    sum += bias[idx];
    output[idx] = sum;
}


__global__ void softmax(float* input){
    float sum=0.0;
    for(int i=0;i<10;i++){
        sum+= exp(input[i]);
        input[i]=exp(input[i]);
    }
    for(int i=0;i<10;i++){
        input[i]/=sum;
    }
}


int main(){
    float conv1_kernel[20*5*5]; float conv1_bias[20];
    float conv2_kernel[50*20*5*5]; float conv2_bias[50];
    float fc1_kernel[500*50*4*4]; float fc1_bias[500];
    float fc2_kernel[10*500]; float fc2_bias[10];

    // float conv1_input[28*28]; 
    // float conv1_output[20*24*24];
    // float pool1_output[20*12*12];
    // float conv2_output[50*8*8];
    // float pool2_output[50*4*4];
    // float fc1_output[500];
    // float fc2_output[10];
    // float soft_output[10];

    // ifstream inputFile1("check.txt");
    // if(!inputFile1.is_open()){std::cerr << "Failed to open the file." << std::endl;return 1;}
    // for(int i=0;i<28*28;i++) inputFile1 >> conv1_input[i];

    ifstream inputFile2("trained_weights/conv1.txt");
    if(!inputFile2.is_open()){std::cerr << "Failed to open the file." << std::endl;return 1;}
    for(int i=0;i<20*5*5;i++) inputFile2 >> conv1_kernel[i];
    for(int i=0;i<20;i++) inputFile2 >> conv1_bias[i];

    ifstream inputFile3("trained_weights/conv2.txt");
    if(!inputFile3.is_open()){std::cerr << "Failed to open the file." << std::endl;return 1;}
    for(int i=0;i<50*20*5*5;i++) inputFile3 >> conv2_kernel[i];
    for(int i=0;i<50;i++) inputFile3 >> conv2_bias[i];

    ifstream inputFile4("trained_weights/fc1.txt");
    if(!inputFile4.is_open()){std::cerr << "Failed to open the file." << std::endl;return 1;}
    for(int i=0;i<500*50*4*4;i++) inputFile4 >> fc1_kernel[i];
    for(int i=0;i<500;i++) inputFile4 >> fc1_bias[i];

    ifstream inputFile5("trained_weights/fc2.txt");
    if(!inputFile5.is_open()){std::cerr << "Failed to open the file." << std::endl;return 1;}
    for(int i=0;i<10*500;i++) inputFile5 >> fc2_kernel[i];
    for(int i=0;i<10;i++) inputFile5 >> fc2_bias[i];


    cudaStream_t stream[num_streams];
    for(int z=0; z<num_streams; z++){
        cudaStreamCreate(&stream[z]);
    }

    // float conv1_input[num_streams][28*28]; 
    // float conv1_output[num_streams][20*24*24];
    // float pool1_output[num_streams][20*12*12];
    // float conv2_output[num_streams][50*8*8];
    // float pool2_output[num_streams][50*4*4];
    // float fc1_output[num_streams][500];
    // float fc2_output[num_streams][10];
    // float soft_output[num_streams][10];

    // Allocate memory for the arrays
    float **conv1_input = new float*[num_streams];
    float **conv1_output = new float*[num_streams];
    float **pool1_output = new float*[num_streams];
    float **conv2_output = new float*[num_streams];
    float **pool2_output = new float*[num_streams];
    float **fc1_output = new float*[num_streams];
    float **fc2_output = new float*[num_streams];
    float **soft_output = new float*[num_streams];

    // Allocate memory for each array
    for (int z = 0; z < num_streams; z++) {
        conv1_input[z] = new float[28*28];
        conv1_output[z] = new float[20*24*24];
        pool1_output[z] = new float[20*12*12];
        conv2_output[z] = new float[50*8*8];
        pool2_output[z] = new float[50*4*4];
        fc1_output[z] = new float[500];
        fc2_output[z] = new float[10];
        soft_output[z] = new float[10];
    }

    float **device_conv1_input= new float*[num_streams];
    float **device_conv1_output= new float*[num_streams];
    float **device_conv1_kernel= new float*[num_streams];
    float **device_conv1_bias= new float*[num_streams];

    for(int z=0; z<num_streams; z++){

        ifstream inputFile1("check.txt");
        if(!inputFile1.is_open()){std::cerr << "Failed to open the file." << std::endl;return 1;}
        for(int i=0;i<28*28;i++) inputFile1 >> conv1_input[z][i];


        // float device_conv1_input[28][28], device_conv1_output[20][24][24], device_conv1_kernel[20][5][5], device_conv1_bias[20];
        // float *device_conv1_input, *device_conv1_output, *device_conv1_kernel, *device_conv1_bias;
        cudaMalloc((void**)&device_conv1_input[z], 28*28*sizeof(float));
        cudaMalloc((void**)&device_conv1_output[z], 20*24*24*sizeof(float));
        cudaMalloc((void**)&device_conv1_kernel[z], 20*5*5*sizeof(float));
        cudaMalloc((void**)&device_conv1_bias[z], 20*sizeof(float));

        cudaMemcpyAsync(device_conv1_input[z], conv1_input[z], 28*28*sizeof(float), cudaMemcpyHostToDevice,stream[z]);
        cudaMemcpyAsync(device_conv1_kernel[z],conv1_kernel, 20*5*5*sizeof(float), cudaMemcpyHostToDevice,stream[z]);
        cudaMemcpyAsync(device_conv1_bias[z], conv1_bias, 20*sizeof(float), cudaMemcpyHostToDevice,stream[z]);

        dim3 conv1_block(24,24,1); dim3 conv1_grid(20,1,1);
        Conv_1<<<conv1_grid, conv1_block,0, stream[z]>>>(device_conv1_input[z], device_conv1_output[z], device_conv1_kernel[z], device_conv1_bias[z]);
        // cudaDeviceSynchronize();
        cudaMemcpyAsync(conv1_output[z],device_conv1_output[z],20*24*24*sizeof(float), cudaMemcpyDeviceToHost,stream[z]);

        cudaFree(device_conv1_input[z]);
        cudaFree(device_conv1_output[z]);
        cudaFree(device_conv1_kernel[z]);
        cudaFree(device_conv1_bias[z]);
    }

    // for(int i=0; i<20*24*24; i++) cout<<conv1_output[0][i]<<endl;

    float **device_pool1_input= new float*[num_streams];
    float **device_pool1_output= new float*[num_streams];

    for(int z=0; z<num_streams; z++){


        // float device_pool1_input[20][24][24], device_pool1_output[20][12][12];
        // float *device_pool1_input, *device_pool1_output;
        cudaMalloc((void**)&device_pool1_input[z], 20*24*24*sizeof(float));
        cudaMalloc((void**)&device_pool1_output[z], 20*12*12*sizeof(float));
        cudaMemcpyAsync(device_pool1_input[z], conv1_output[z], 20*24*24*sizeof(float), cudaMemcpyHostToDevice,stream[z]);
        dim3 pool1_block(12,12,1); dim3 pool1_grid(20,1,1);
        Pool_1<<<pool1_grid, pool1_block,0, stream[z]>>>(device_pool1_input[z], device_pool1_output[z]);
        // cudaDeviceSynchronize();
        cudaMemcpyAsync(pool1_output[z], device_pool1_output[z], 20*12*12*sizeof(float), cudaMemcpyDeviceToHost,stream[z]);

        cudaFree(device_pool1_input[z]);
        cudaFree(device_pool1_output[z]);
    }

    // for(int i=0; i<20*12*12;i++) cout<<pool1_output[0][i]<<endl;

    float **device_conv2_input= new float*[num_streams];
    float **device_conv2_output= new float*[num_streams];
    float **device_conv2_kernel= new float*[num_streams];
    float **device_conv2_bias= new float*[num_streams];

    for(int z=0; z<num_streams; z++){


        // float device_conv2_input[20][12][12], device_conv2_output[50][8][8], device_conv2_kernel[50][20][5][5], device_conv2_bias[50];
        // float *device_conv2_input, *device_conv2_output, *device_conv2_kernel, *device_conv2_bias;
        cudaMalloc((void**)&device_conv2_input[z], 20*12*12*sizeof(float));
        cudaMalloc((void**)&device_conv2_output[z], 50*8*8*sizeof(float));
        cudaMalloc((void**)&device_conv2_kernel[z], 50*20*5*5*sizeof(float));
        cudaMalloc((void**)&device_conv2_bias[z], 50*sizeof(float));

        cudaMemcpyAsync(device_conv2_input[z], pool1_output[z], 20*12*12*sizeof(float), cudaMemcpyHostToDevice,stream[z]);
        cudaMemcpyAsync(device_conv2_kernel[z], conv2_kernel, 50*20*5*5*sizeof(float), cudaMemcpyHostToDevice,stream[z]);
        cudaMemcpyAsync(device_conv2_bias[z], conv2_bias, 50*sizeof(float), cudaMemcpyHostToDevice);

        dim3 conv2_block(8,8,1); dim3 conv2_grid(50,1,1);
        Conv_2<<<conv2_grid, conv2_block,0, stream[z]>>>(device_conv2_input[z], device_conv2_output[z], device_conv2_kernel[z], device_conv2_bias[z]);
        // cudaDeviceSynchronize();
        cudaMemcpyAsync(conv2_output[z], device_conv2_output[z], 50*8*8*sizeof(float), cudaMemcpyDeviceToHost,stream[z]);

        cudaFree(device_conv2_input[z]);
        cudaFree(device_conv2_output[z]);
        cudaFree(device_conv2_kernel[z]);
        cudaFree(device_conv2_bias[z]);
    }

    // for(int i=0;i<50*8*8;i++) cout<<conv2_output[0][i]<<endl;


    float **device_pool2_input= new float*[num_streams];
    float **device_pool2_output= new float*[num_streams];

    for(int z=0; z<num_streams; z++){

        // float device_pool2_input[50][8][8], device_pool2_output[50][4][4];
        // float *device_pool2_input, *device_pool2_output;
        cudaMalloc((void**)&device_pool2_input[z], 50*8*8*sizeof(float));
        cudaMalloc((void**)&device_pool2_output[z], 50*4*4*sizeof(float));
        cudaMemcpyAsync(device_pool2_input[z], conv2_output[z], 50*8*8*sizeof(float), cudaMemcpyHostToDevice,stream[z]);
        dim3 pool2_block(4,4,1); dim3 pool2_grid(50,1,1);
        Pool_2<<<pool2_grid, pool2_block,0, stream[z]>>>(device_pool2_input[z], device_pool2_output[z]);
        // cudaDeviceSynchronize();
        cudaMemcpyAsync(pool2_output[z], device_pool2_output[z], 50*4*4*sizeof(float), cudaMemcpyDeviceToHost,stream[z]);

        cudaFree(device_pool2_input[z]);
        cudaFree(device_pool2_output[z]);
    }

    // for(int i=0; i<50*4*4;i++) cout<<pool2_output[0][i]<<endl;

    float **device_fc1_input= new float*[num_streams];
    float **device_fc1_output= new float*[num_streams];
    float **device_fc1_kernel= new float*[num_streams];
    float **device_fc1_bias= new float*[num_streams];

    for(int z=0; z<num_streams; z++){

        // float device_fc1_input[50][4][4], device_fc1_output[500], device_fc1_kernel[500][50][4][4], device_fc1_bias[50];
        // float *device_fc1_input, *device_fc1_output, *device_fc1_kernel, *device_fc1_bias;
        cudaMalloc((void**)&device_fc1_input[z], 50*4*4*sizeof(float));
        cudaMalloc((void**)&device_fc1_output[z], 500*sizeof(float));
        cudaMalloc((void**)&device_fc1_kernel[z], 500*50*4*4*sizeof(float));
        cudaMalloc((void**)&device_fc1_bias[z], 500*sizeof(float));

        cudaMemcpyAsync(device_fc1_input[z], pool2_output[z], 50*4*4*sizeof(float), cudaMemcpyHostToDevice,stream[z]);
        cudaMemcpyAsync(device_fc1_kernel[z], fc1_kernel, 500*50*4*4*sizeof(float), cudaMemcpyHostToDevice,stream[z]);
        cudaMemcpyAsync(device_fc1_bias[z], fc1_bias, 500*sizeof(float), cudaMemcpyHostToDevice,stream[z]);

        dim3 fc1_block(500,1,1); dim3 fc1_grid(1,1,1);
        FC_1<<<fc1_grid, fc1_block,0, stream[z]>>>(device_fc1_input[z], device_fc1_output[z], device_fc1_kernel[z], device_fc1_bias[z]);
        // cudaDeviceSynchronize();
        cudaMemcpyAsync(fc1_output[z], device_fc1_output[z], 500*sizeof(float), cudaMemcpyDeviceToHost,stream[z]);

        cudaFree(device_fc1_input[z]);
        cudaFree(device_fc1_output[z]);
        cudaFree(device_fc1_kernel[z]);
        cudaFree(device_fc1_bias[z]);
    }

    float **device_fc2_input= new float*[num_streams];
    float **device_fc2_output= new float*[num_streams];
    float **device_fc2_kernel= new float*[num_streams];
    float **device_fc2_bias= new float*[num_streams];

    for(int z=0; z<num_streams; z++){

        // float device_fc2_input[500], device_fc2_output[10], device_fc2_kernel[10][500], device_fc2_bias[10];
        // float *device_fc2_input, *device_fc2_output, *device_fc2_kernel, *device_fc2_bias;
        cudaMalloc((void**)&device_fc2_input[z], 500*sizeof(float));
        cudaMalloc((void**)&device_fc2_output[z], 10*sizeof(float));
        cudaMalloc((void**)&device_fc2_kernel[z], 10*500*sizeof(float));
        cudaMalloc((void**)&device_fc2_bias[z], 10*sizeof(float));

        cudaMemcpyAsync(device_fc2_input[z], fc1_output[z], 500*sizeof(float), cudaMemcpyHostToDevice,stream[z]);
        cudaMemcpyAsync(device_fc2_kernel[z], fc2_kernel, 10*500*sizeof(float), cudaMemcpyHostToDevice,stream[z]);
        cudaMemcpyAsync(device_fc2_bias[z], fc2_bias, 10*sizeof(float), cudaMemcpyHostToDevice,stream[z]);

        dim3 fc2_block(10,1,1); dim3 fc2_grid(1,1,1);
        FC_2<<<fc2_grid, fc2_block,0, stream[z]>>>(device_fc2_input[z], device_fc2_output[z], device_fc2_kernel[z], device_fc2_bias[z]);
        // cudaDeviceSynchronize();
        cudaMemcpyAsync(fc2_output[z], device_fc2_output[z], 10*sizeof(float), cudaMemcpyDeviceToHost,stream[z]);

        cudaFree(device_fc2_input[z]);
        cudaFree(device_fc2_output[z]);
        cudaFree(device_fc2_kernel[z]);
        cudaFree(device_fc2_bias[z]);
    }

    float **device_soft_input= new float*[num_streams];

    for(int z=0; z<num_streams; z++){

        // float device_soft_input[10], device_soft_output[10];
        // float *device_soft_input;
        cudaMalloc((void**)&device_soft_input[z], 10*sizeof(float));

        cudaMemcpyAsync(device_soft_input[z], fc2_output[z], 10*sizeof(float), cudaMemcpyHostToDevice,stream[z]);
        dim3 soft_block(1,1,1); dim3 soft_grid(1,1,1); 
        softmax<<<soft_grid, soft_block,0, stream[z]>>>(device_soft_input[z]);
        // cudaDeviceSynchronize();
        cudaMemcpyAsync(soft_output[z], device_soft_input[z], 10*sizeof(float), cudaMemcpyDeviceToHost,stream[z]);

        // for(int i=0;i<10;i++){
        //     cout<<soft_output[i]<<endl;
        // }

        vector<pair<float, int> > indexed_values;

        for (int i = 0; i < 10; ++i) {
            indexed_values.push_back(make_pair(-1.0*soft_output[z][i],i));
        }
        sort(indexed_values.begin(),indexed_values.end());
        for (int i = 0; i < 5; ++i) {
            cout << -100.0*indexed_values[i].first << " class " << indexed_values[i].second << endl;
        }

        cudaFree(device_soft_input[z]);
    }

    for(int z=0; z<num_streams; z++){
        cudaStreamDestroy(stream[z]);
    }
}