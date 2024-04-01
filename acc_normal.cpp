#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <dirent.h>
using namespace std;

void Conv_1(float* input, float* output, float* kernel, float* bias){
    for(int a=0; a<20; a++){
        for(int b=0; b<24; b++){
            for(int c=0; c<24; c++){
                float sum=0.0;
                for(int i=0;i<5;i++){
                    for(int j=0;j<5;j++){
                        sum+=input[(b+i)*28 +(c+j)]* kernel[a*5*5+i*5+j];
                    }
                }
                sum+=bias[a];
                output[a*24*24+b*24+c]=sum;
            }
        }
    }
}

void Pool_1(float* input, float* output){
    for(int a=0;a<20;a++){
        for(int b=0;b<12;b++){
            for(int c=0;c<12;c++){
                float val= input[a*24*24 + (2*b)*24 + (2*c)];
                val= fmaxf(val, input[a*24*24 + (2*b+1)*24 + (2*c)]);
                val= fmaxf(val, input[a*24*24 + (2*b+1)*24 + (2*c+1)]);
                val= fmaxf(val, input[a*24*24 + (2*b)*24 + (2*c+1)]);
                output[a*12*12+ b*12 +c]= val;
            }
        }
    }
}

void Conv_2(float* input, float* output, float* kernel, float* bias){
    for(int a=0;a<50;a++){
        for(int b=0;b<8;b++){
            for(int c=0;c<8;c++){
                float sum=0.0;
                for(int i=0;i<20;i++){
                    for(int j=0;j<5;j++){
                        for(int k=0;k<5;k++){
                            sum+= input[i*12*12 + (b+j)*12 + (c+k)]*kernel[a*20*5*5 + i*5*5 + j*5 +k];
                        }
                    }
                }
                sum+= bias[a];
                output[a*8*8 + b*8 + c]=sum;
            }
        }
    }
}

void Pool_2(float* input, float* output){
    for(int a=0;a<50;a++){
        for(int b=0;b<4;b++){
            for(int c=0;c<4;c++){
                float val= input[a*8*8 + (2*b)*8 + (2*c)];
                val= fmaxf(val, input[a*8*8 + (2*b+1)*8 + (2*c)]);
                val= fmaxf(val, input[a*8*8 + (2*b+1)*8 + (2*c+1)]);
                val= fmaxf(val, input[a*8*8 + (2*b)*8 + (2*c+1)]);
                output[a*4*4+ b*4 +c]= val;
            }
        }
    }
}

void FC_1(float* input, float* output, float* kernel, float* bias){
    for(int a=0;a<500;a++){
        float sum=0.0;
        for(int i=0;i<50;i++){
            for(int j=0;j<4;j++){
                for(int k=0;k<4;k++){
                    sum+= input[i*4*4 + j*4 + k]*kernel[a*50*4*4 + i*4*4 + j*4 + k];
                }
            }
        }
        sum+=bias[a];
        output[a]=fmaxf(0.0,sum);
    }
}

void FC_2(float* input, float* output, float* kernel, float* bias){
    for(int a=0;a<10;a++){
        float sum=0.0;
        for(int i=0;i<500;i++){
            sum+= input[i]*kernel[a*500+i];
        }
        output[a]=sum;
    }
}


int main(){
    int right=0; int wrong=0;

    float conv1_kernel[20*5*5]; float conv1_bias[20];
    float conv2_kernel[50*20*5*5]; float conv2_bias[50];
    float fc1_kernel[500*50*4*4]; float fc1_bias[500];
    float fc2_kernel[10*500]; float fc2_bias[10];

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

    struct dirent *ent;
    const char *inputdir = "images/";
    DIR *dir = opendir(inputdir);

    while ((ent = readdir(dir)) != NULL) {
        float conv1_input[28*28]; 
        float conv1_output[20*24*24];
        float pool1_output[20*12*12];
        float conv2_output[50*8*8];
        float pool2_output[50*4*4];
        float fc1_output[500];
        float fc2_output[10];
        float soft_output[10];
        string filename = ent->d_name;
        int ans= filename[filename.size()-5]-'0';
        ifstream inputFile1(inputdir+filename);
        if(!inputFile1.is_open()){std::cerr << "Failed to open the file." << std::endl;return 1;}
        for(int i=0;i<28*28;i++) inputFile1 >> conv1_input[i];

        Conv_1(conv1_input, conv1_output, conv1_kernel, conv1_bias);
        Pool_1(conv1_output, pool1_output);
        Conv_2(pool1_output, conv2_output, conv2_kernel, conv2_bias);
        Pool_2(conv2_output, pool2_output);

        FC_1(pool2_output, fc1_output, fc1_kernel, fc1_bias);
        FC_2(fc1_output, fc2_output, fc2_kernel, fc2_bias);
        
        float sum=0.0;
        for(int i=0;i<10;i++) sum+=exp(fc2_output[i]);
        for(int i=0;i<10;i++) soft_output[i]= exp(fc2_output[i])/sum;

        vector<pair<float, int> > indexed_values;

        for (int i = 0; i < 10; ++i) {
            indexed_values.push_back(make_pair(-1.0*soft_output[i],i));
        }
        sort(indexed_values.begin(),indexed_values.end());
        // for (int i = 0; i < 5; ++i) {
        //     cout << -100.0*indexed_values[i].first << " class " << indexed_values[i].second << endl;
        // }
        // cout<<ans<<"ans"<<indexed_values[0].second<<endl;
        if(ans==indexed_values[0].second){right++;}
        else wrong++;
    }
    closedir(dir);
    cout<<right<<endl;
    
    cout<<wrong<<endl;

}