//
//  SBPNN.cpp
//  SBPNN
//
//  Created by Jerry Zhang on 29/03/2018.
//  Copyright © 2018 Jerry Zhang. All rights reserved.
//
#include "SBPNN.hpp"
#include <iostream>
using std::endl;
bool test(SBPNN *net, double *input_unit,int input_num,double *target,int target_num) {
    net->input_units[0] = 1.0;
    for(int i = 1;i <= input_num;i++) {
        net->input_units[i] = input_unit[i - 1];
    }
    for(int i = 0;i <=  target_num;i++) {
        net->target[i] = target[i];
    }
    bool result = true;
    layerforward(net->input_units, net->hidden_units, net->input_weights, net->input_n, net->hidden_n);
    layerforward(net->hidden_units, net->output_units, net->hidden_weights, net->hidden_n, net->output_n);
    for(int i = 1;i <= net->output_n;i++) {
        if(fabs(net->output_units[i] - net->target[i]) > 0.4) {
            result = false;
        }
    }
    if(result == false) {
        std::cout<<"Testing failure"<<std::endl;
        std::cout<<"Expecting:" <<std::endl;
        for(int i = 1;i < net->output_n;i++)
            std::cout<<net->target[i]<<" ";
        std::cout<<endl;
        std::cout<<"But got:"<<endl;
        for(int i = 0;i < net->output_n;i++) {
            std::cout<<net->output_units[i]<<" ";
        }
        std::cout<<endl;
        
    }
    return result;
    
}
void train(SBPNN *net, double *input_unit,int input_num, double *target,int target_num, double *eo, double *eh) {
    
    for(int i = 1;i < input_num + 1;i++) {
        net->input_units[i] = input_unit[i - 1];
    }
    for(int i = 1;i < target_num + 1;i++) {
        net->target[i] = target[i - 1];
    }
    
    layerforward(net->input_units, net->hidden_units, net->input_weights, net->input_n, net->hidden_n);
    layerforward(net->hidden_units, net->output_units, net->hidden_weights, net->hidden_n, net->output_n);
    getOutputError(net->output_delta, net->target, net->output_units, net->output_n, eo);
    getHiddenError(net->hidden_delta, net->hidden_n, net->output_delta, net->output_n, net->hidden_weights, net->hidden_units, eh);
    
    adjustWeights(net->output_delta, net->output_n, net->hidden_units, net->hidden_n, net->hidden_weights, net->hidden_prev_weights, net->eta, net->momentum);
    adjustWeights(net->hidden_delta, net->hidden_n, net->input_units, net->input_n, net->input_weights, net->input_prev_weights, net->momentum, net->eta);
}
void adjustWeights(double *delta, int ndelta, double *ly, int nly, double** w, double **oldw, double eta, double momentum) {
//    hidden_weights = hidden_prev_weights + eta*output_delta*hidden_units + momentum*hidden_prev_weights
    //input_weights = input_prev_weights + eta*hidden_delta*input_units + momentum*input_prev_weights
    
    for(int i = 0;i <= nly;i++) {
        for(int j = 0;j <= ndelta;j++) {
            w[i][j] = w[i][j] - eta * delta[j] * ly[i] + momentum * oldw[i][j];
            oldw[i][j] = - eta * delta[j] * ly[i] + momentum * oldw[i][j];
        }
    }
}
    //adjustWeights(net->hidden_delta, net->hidden_n, net->input_units, net->input_n, net->input_weights, net->input_prev_weights, net->momentum, net->eta);

double sigmoidal(double x) {
    return 1 / (1 + exp(-x));
}
//void layerforward(double *l1, double *l2, double **conn, int n1, int n2);
void layerforward(double *l1, double *l2, double **conn, int n1, int n2) {
    for(int j = 1;j <= n2;j++) {
        l2[j] = 0;
        for(int i = 0;i <= n1;i++)
            l2[j] += l1[i] * conn[i][j];
        l2[j] = sigmoidal(l2[j]);
    }
}

// TODO: Refactor the following two functions.
void getOutputError(double *delta, double *target, double *output, int nj, double *err) {
    *err = 0;
    delta[0] = 0;
    for(int i = 1;i <= nj;i++) {
        delta[i] = -(target[i] - output[i])*output[i] * (1 - output[i]);
        *err += fabs(target[i] - output[i]);
//        std::cout<<"target ="<<target[i]<<" output="<<output[i]<<endl;
//        std::cout<<"err +="<<fabs(target[i] - output[i])<<endl;
    }
}


void getHiddenError(double* delta_h, int nh, double *delta_o, int no, double **who, double *hidden, double *err) {
    for(int i = 0; i <= nh;i++) {
        double sum = 0;
        for(int j = 0;j <= no;j++)
            sum += delta_o[j] * who[i][j];
        delta_h[i] = sum * hidden[i] * (1 - hidden[i]);
    }
    
}

void freeBPNN(SBPNN * net) {
    delete [] net->input_units;
    delete [] net->hidden_units;
    delete [] net->output_units;
    delete [] net->target;
    delete [] net->hidden_delta;
    delete [] net->output_delta;
    for(int i = 0;i < net->input_n + 1;i++) {
        delete [] net->input_weights[i];
    }
    delete [] net->input_weights;
    
    for(int i = 0;i < net->hidden_n + 1;i++) {
        delete [] net->hidden_weights[i];
    }
    delete [] net->hidden_weights;
    
    
    for(int i = 0;i < net->input_n + 1;i++) {
        delete [] net->input_prev_weights[i];
    }
    delete [] net->input_prev_weights;

    for(int i = 0;i < net->hidden_n + 1;i++) {
        delete [] net->hidden_prev_weights[i];
    }
    delete [] net->hidden_prev_weights;
    delete net;
}

void saveBPNN(SBPNN * net, char *filename) {
    std::ofstream file(filename);
    while(!file.is_open()) {
        std::cout<<"file open failure, enter a new file name"<<endl;
        std::cin>> filename;
        file.open(filename);
    }
    if(file.is_open()) {
        file<<net->input_n<<endl;
        file<<net->hidden_n<<endl;
        file<<net->output_n<<endl;
        for(int i = 0;i < net->input_n + 1;i++)
            for(int j = 0;j <= net->hidden_n;j++) {
                file<<net->input_weights[i][j]<<endl;
            }
        
        for(int i = 0;i < net->hidden_n + 1;i++)
            for(int j = 0;j <= net->output_n;j++) {
                file<<net->hidden_weights[i][j]<<endl;
            }
    }
    file.close();
}

// TODO: Needs refactor.
SBPNN* readBPNN(SBPNN * net, char* filename) {
    std::ifstream file(filename);
    if(file.is_open()) {
        int intput_num,hidden_num,output_num;
        file>>intput_num;
        file>>hidden_num;
        file>>output_num;
        freeBPNN(net);
        net = createBPNN(intput_num, hidden_num, output_num);
        for(int i = 0;i < net->input_n + 1;i++)
            for(int j = 0;j <= net->hidden_n;j++) {
                file>>net->input_weights[i][j];
            }
//        std::cout<<net->input_weights[0][0]<<endl;
//        exit(0);

        for(int i = 0;i < net->hidden_n + 1;i++)
            for(int j = 0;j <= net->output_n;j++) {
                file>>net->hidden_weights[i][j];
            }

        file.close();
    }
    else {
        std::cerr<<"Open file failure!"<<endl;
    }
    return net;

}


void initSeed(int seed = (int)time(NULL)) {
    srand(seed);
}



double fRand(double fMin, double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}


SBPNN * createBPNN(int n_in,int n_hidden,int n_out) {
    initSeed();
    SBPNN * bpnn = new SBPNN;
    bpnn->input_n = n_in;
    bpnn->hidden_n = n_hidden;
    bpnn->output_n = n_out;
    bpnn->input_units   = new double[bpnn->input_n + 1];
    bpnn->input_units[0] = 1.0;
    bpnn->hidden_units  = new double[bpnn->hidden_n + 1];
    bpnn->hidden_units[0] = 1.0;
    bpnn->output_units  = new double[bpnn->output_n + 1];
    bpnn->target        = new double[bpnn->output_n + 1];
    bpnn->hidden_delta  = new double[bpnn->hidden_n + 1];
    bpnn->output_delta  = new double[bpnn->output_n + 1];
    bpnn->input_weights = new double*[bpnn->input_n + 1];
    for(int i = 0;i < bpnn->input_n + 1;i++) {
        bpnn->input_weights[i] = new double[bpnn->hidden_n + 1];
        for(int j = 0;j <= bpnn->hidden_n;j++) {
            bpnn->input_weights[i][j] = fRand(-0.05, 0.05);
        }
    }
    
    bpnn->hidden_weights = new double*[bpnn->hidden_n + 1];
    for(int i = 0;i < bpnn->hidden_n + 1;i++) {
        bpnn->hidden_weights[i] = new double[bpnn->output_n + 1];
        for(int j = 0;j <= bpnn->output_n;j++) {
            bpnn->hidden_weights[i][j] = fRand(-0.05, 0.05);
        }
    }

    bpnn->input_prev_weights = new double*[bpnn->input_n + 1];
    for(int i = 0;i < bpnn->input_n + 1;i++) {
        bpnn->input_prev_weights[i] = new double[bpnn->hidden_n + 1];
        for(int j = 0;j <= bpnn->hidden_n;j++) {
            bpnn->input_prev_weights[i][j] = 0;
        }
    }
    bpnn->hidden_prev_weights = new double*[bpnn->hidden_n + 1];
    for(int i = 0;i < bpnn->hidden_n + 1;i++) {
        bpnn->hidden_prev_weights[i] = new double[bpnn->output_n + 1];
        for(int j = 0;j <= bpnn->output_n;j++) {
            bpnn->hidden_prev_weights[i][j] = 0;
        }
    }
    bpnn->eta = 0.3;
    bpnn->momentum = 0.3;
    std::cout<<"Create SBNPP success."<<endl;
    return bpnn;
}
