#include <iostream>

__global__ void Conv_1(float input[28][28], float output[20][24][24], float kernel[20][5][5], float bias[20]){
    int channel= blockId.x;
    int a= threadId.x; int b= threadId.y;
    float sum=0.0;
    for(int i=0;i<5;i++){
        for(int j=0;j<5;j++){
            int x=i+a; int y=j+b;
            sum+=kernel[channel][i][j]*input[x][y];
        }
    }
    sum+=bias[channel];
    output[channel][a][b]=sum;
}

__global__ void Pool_1(float input[20][24][24], float output[20][12][12]){
    int channel= blockId.x;
    int a= threadId.x; int b= threadId.y;
    output[channel][a][b]= max({input[channel][2*a][2*b],input[channel][2*a+1][2*b],input[channel][2*a][2*b+1],input[channel][2*a+1][2*b+1]});
}

__global__ void Conv_2(float input[20][12][12], float output[50][8][8], float kernel[50][5][5][20], float bias[50]){
    int chx= blockId.x; 
    int thx= threadId.x; int thy= threadId.y;
    float sum=0.0;
    for(int c=0; c<20; c++){
        for(int i=0;i<5;i++){
            for(int j=0;j<5;j++){
                int x=i+thx; int y=j+thy;
                sum+=kernel[chx][i][j][c]*input[c][x][y];
            }
        }
    }
    sum+=bias[chx];
    output[chx][thx][thy]=sum;
}

__global__ void Pool_2(float input[50][8][8], float output[50][4][4]){
    int channel= blockId.x;
    int a= threadId.x; int b= threadId.y;
    output[channel][a][b]= max({input[channel][2*a][2*b],input[channel][2*a+1][2*b],input[channel][2*a][2*b+1],input[channel][2*a+1][2*b+1]});
}

__global__ void FC_1(float input[50][4][4], float output[500], float kernel[500][4][4][50], float bias[500]){
    int idx= threadId.x;
    float sum=0.0;
    for (int c = 0; c < 50; ++c) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                sum += kernel[idx][i][j][c] * input[c][i][j];
            }
        }
    }
    sum += bias[idx];
    output[idx] = max(0.0,sum);
}


__global__ void FC_2(float input[500], float output[10], float kernel[10][500], float bias[10]){
    int idx= threadId.x;
    float sum=0.0;
    for(int i=0;i<500;i++){
        sum+=kernel[idx][i]*input[i];
    }
    sum+=bias[idx];
    output[idx]=sum;
}

__global__ void softmax(float input[10]){
    float sum=0.0;
    for(int i=0;i<10;i++){
        sum+= exp(input[i]);
    }
    for(int i=0;i<10;i++){
        input[i]/=sum;
    }
}


int main(){
    float conv1_kernel[20][5][5]; float conv1_bias[20];
    float conv2_kernel[50][5][5][20]; float conv2_bias[50];
    float fc1_kernel[500][4][4][50]; float fc1_bias[500];
    float fc2_kernel[10][500]; float fc2_bias[10];



    float conv1_input[28][28]; 
    // float conv1_output[20][24][24];
    // float pool1_output[20][12][12];
    // float conv2_output[50][8][8];
    // float pool2_output[50][4][4];
    // float fc1_output[500];
    float fc2_output[10];



    float *device_conv1_input, *device_conv1_output, *device_conv1_kernel, *device_conv1_bias;
    cudaMalloc((void**)&device_conv1_input, 28*28*sizeof(float));
    cudaMalloc((void**)&device_conv1_output, 20*24*24*sizeof(float));
    cudaMalloc((void**)&device_conv1_kernel, 20*5*5*sizeof(float));
    cudaMalloc((void**)&device_conv1_bias, 20*sizeof(float));

    cudaMemcpy(device_conv1_input, conv1_input, 28*28*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_conv1_kernel, conv1_kernel, 20*5*5*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_conv1_bias, conv1_bias, 20*sizeof(float), cudaMemcpyHostToDevice);

    dim3 conv1_block(24,24,1); dim3 conv1_grid(20,1,1);
    Conv_1<<<conv1_grid, conv1_block>>>(device_conv1_input, device_conv1_output, device_conv1_kernel, device_conv1_bias);
    cudaDeviceSynchronize();



    float *device_pool1_output;
    cudaMalloc((void**)&device_pool1_output, 20*12*12*sizeof(float));
    dim3 pool1_block(12,12,1); dim3 pool1_grid(20,1,1);
    Pool_1<<<pool1_grid, pool1_block>>>(device_conv1_output, device_pool1_output);
    cudaDeviceSynchronize();



    float *device_conv2_output, *device_conv2_kernel, *device_conv2_bias;
    cudaMalloc((void**)&device_conv2_output, 50*8*8*sizeof(float));
    cudaMalloc((void**)&device_conv2_kernel, 50*5*5*20*sizeof(float));
    cudaMalloc((void**)&device_conv2_bias, 50*sizeof(float));

    cudaMemcpy(device_conv2_kernel, conv2_kernel, 50*5*5*20*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_conv2_bias, conv2_bias, 50*sizeof(float), cudaMemcpyHostToDevice);

    dim3 conv2_block(8,8,1); dim3 conv2_grid(50,1,1);
    Conv_2<<<conv2_grid, conv2_block>>>(device_pool1_output, device_conv2_output, device_conv2_kernel, device_conv2_bias);
    cudaDeviceSynchronize();



    float *device_pool2_output;
    cudaMalloc((void**)&device_pool2_output, 50*4*4*sizeof(float));
    dim3 pool2_block(4,4,1); dim3 pool2_grid(50,1,1);
    Pool_2<<<pool2_grid, pool2_block>>>(device_conv2_output, device_pool2_output);
    cudaDeviceSynchronize();



    float *device_fc1_output, *device_fc1_kernel, *device_fc1_bias;
    cudaMalloc((void**)&device_fc1_output, 500*sizeof(float));
    cudaMalloc((void**)&device_fc1_kernel, 500*4*4*50*sizeof(float));
    cudaMalloc((void**)&device_fc1_bias, 500*sizeof(float));

    cudaMemcpy(device_fc1_kernel, fc1_kernel, 500*4*4*50*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_fc1_bias, fc1_bias, 500*sizeof(float), cudaMemcpyHostToDevice);

    dim3 fc1_block(500,1,1); dim3 fc1_grid(1,1,1);
    FC_1<<<fc1_grid, fc1_block>>>(device_pool2_output, device_fc1_output, device_fc1_kernel, device_fc1_bias);
    cudaDeviceSynchronize();



    float *device_fc2_output, *device_fc2_kernel, *device_fc2_bias;
    cudaMalloc((void**)&device_fc2_output, 10*sizeof(float));
    cudaMalloc((void**)&device_fc2_kernel, 10*50*sizeof(float));
    cudaMalloc((void**)&device_fc2_bias, 10*sizeof(float));

    cudaMemcpy(device_fc2_kernel, fc2_kernel, 10*50*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_fc2_bias, fc2_bias, 10*sizeof(float), cudaMemcpyHostToDevice);

    dim3 fc2_block(10,1,1); dim3 fc2_grid(1,1,1);
    FC_2<<<fc2_grid, fc2_block>>>(device_pool2_output, device_fc2_output, device_fc2_kernel, device_fc2_bias);
    cudaDeviceSynchronize();



    dim3 soft_block(1,1,1); dim3 soft_grid(1,1,1); 
    softmax<<<soft_grid, soft_block>>>(device_fc2_output);
    cudaDeviceSynchronize();
    cudaMemcpy(fc2_output, device_fc2_output, 10*sizeof(float), cudaMemcpyDeviceToHost);
}