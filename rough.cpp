#include<iostream>
#include<vector>
#include<utility>
#include<cmath>
#include<algorithm>
#include<set>
#include<map>
#include<bitset>
#include<fstream>
using namespace std;


int main(){
    float conv1_kernel[20][5][5]; float conv1_bias[20];
    float conv2_kernel[50][5][5][20]; float conv2_bias[50];
    float fc1_kernel[500][4][4][50]; float fc1_bias[500];
    float fc2_kernel[10][500]; float fc2_bias[10];

    float input[28][28];

    ifstream input_file("input.txt");
    if (!input_file.is_open()) {
        cerr << "Error: Unable to open weights file." << endl;
        return 1;
    }

    // Read weights from the file into conv1_kernel array
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            
            if (!(input_file >> input[i][j])) {
                cerr << "Error: Unable to read weights from file." << endl;
                break;
            }
            
        }
    }

    input_file.close();



    ifstream conv1_file("trained_weights/conv1.txt");
    if (!conv1_file.is_open()) {
        cerr << "Error: Unable to open weights file." << endl;
        return 1;
    }

    // Read weights from the file into conv1_kernel array
    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 5; ++j) {
            for (int k = 0; k < 5; ++k) {
                if (!(conv1_file >> conv1_kernel[i][j][k])) {
                    cerr << "Error: Unable to read weights from file." << endl;
                    break;
                }
            }
        }
    }

    for(int i=0;i<20;i++){
        if (!(conv1_file >> conv1_bias[i])) {
            cerr << "Error: Unable to read weights from file." << endl;
            break;
        }
    }

    conv1_file.close();


    ifstream conv2_file("trained_weights/conv2.txt");
    if (!conv2_file.is_open()) {
        cerr << "Error: Unable to open conv2 weights file." << endl;
        return 1;
    }
    // Read weights from the file into conv2_kernel array
    // (Assuming weights are stored in row-major order)
    for (int i = 0; i < 50; ++i) {
        for (int j = 0; j < 5; ++j) {
            for (int k = 0; k < 5; ++k) {
                for (int l = 0; l < 20; ++l) {
                    if (!(conv2_file >> conv2_kernel[i][j][k][l])) {
                        cerr << "Error: Unable to read conv2 weights from file." << endl;
                        return 1;
                    }
                }
            }
        }
    }

    for(int i=0;i<50;i++){
        if (!(conv2_file >> conv2_bias[i])) {
            cerr << "Error: Unable to read weights from file." << endl;
            break;
        }
    }

    conv2_file.close();


    ifstream fc1_file("trained_weights/fc1.txt");
    if (!fc1_file.is_open()) {
        cerr << "Error: Unable to open fc1 weights file." << endl;
        return 1;
    }
    // Read weights from the file into fc1_kernel array
    // (Assuming weights are stored in row-major order)
    for (int i = 0; i < 500; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                for (int l = 0; l < 50; ++l) {
                    if (!(fc1_file >> fc1_kernel[i][j][k][l])) {
                        cerr << "Error: Unable to read fc1 weights from file." << endl;
                        return 1;
                    }
                }
            }
        }
    }

    for(int i=0;i<500;i++){
        if (!(fc1_file >> fc1_bias[i])) {
            cerr << "Error: Unable to read weights from file." << endl;
            break;
        }
    }

    fc1_file.close();


    std::ifstream fc2_file("trained_weights/fc2.txt");
    if (!fc2_file.is_open()) {
        std::cerr << "Error: Unable to open fc2 weights file." << std::endl;
        return 1;
    }
    // Read weights from the file into fc2_kernel array
    // (Assuming weights are stored in row-major order)
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 500; ++j) {
            if (!(fc2_file >> fc2_kernel[i][j])) {
                std::cerr << "Error: Unable to read fc2 weights from file." << std::endl;
                return 1;
            }
        }
    }

    for(int i=0;i<10;i++){
        if (!(fc2_file >> fc2_bias[i])) {
            cerr << "Error: Unable to read weights from file." << endl;
            break;
        }
    }

    fc2_file.close();














     for(int i=0;i<10;i++){
        cout<<fc2_bias[i]<<endl;
    }

}