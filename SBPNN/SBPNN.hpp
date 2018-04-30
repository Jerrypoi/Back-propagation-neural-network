//
//  SBPNN.hpp
//  SBPNN
//
//  Created by Jerry Zhang on 29/03/2018.
//  Copyright © 2018 Jerry Zhang. All rights reserved.
//

#ifndef SBPNN_hpp
#define SBPNN_hpp

#include <cstdlib>
#include <fstream>
#include <ctime>
#include <cmath>
struct SBPNN {
    int input_n;                  // 输入层的神经元个数
    int hidden_n;                 // 隐含层的神经元个数
    int output_n;                 // 输出层的神经元个数
    double *input_units;          // 输入层神经元的输出(input_units[i], i=1,2,...,input_n),其中input_units[0]=1.0
    double *hidden_units;         // 隐藏层神经元的输出(hidden_units[i], i=1,2,...,hidden_n),其中hidden_units[0]=1.0
    double *output_units;         // 输出层神经元的输出(output_units[i], i=1,2,...,output_n)
    double *hidden_delta;        // 隐藏层神经元的学习误差(hidden_delta[i],i=1,...,hidden_n)
    double *output_delta;         // 输出层神经元的学习误差(output_delta[i],i=1,...,output_n)
    double *target;               // 目标向量(target[i],i=1,...,output_n)
    double **input_weights;
    // 输入层到隐藏层的连接加权系数 input_weights[i][j]表示第i个输入层与第j个隐藏层神经元的加权系数,input_weights[0][j]表示隐藏层第j个神经元的阈值
    double **hidden_weights;
    // 隐藏层到输出层的连接加权系数 hidden_weights[i][j]表示第i个隐藏层与第j个输出层神经元的加权系数,hidden_weights[0][j]表示输出层第j个神经元的阈值
    // 下面两个在迭代时使用
    double **input_prev_weights;  // 前次输入层到隐藏层权值的改变
    double **hidden_prev_weights; // 前次隐藏层到输出层权值的改变
    double eta;                   // 学习速率,初始值0.3, hidden_weights = hidden_prev_weights + eta*output_delta*hidden_units + momentum*hidden_prev_weights
    double momentum;              // 动量系数,初始值0.3, input_weights = input_prev_weights + eta*hidden_delta*input_units + momentum*input_prev_weights
};
double fRand(double fMin, double fMax);
void initSeed(int seed);
SBPNN* readBPNN(SBPNN *, char* filename);
void saveBPNN(SBPNN *, char *filename);
SBPNN* createBPNN(int n_in,int n_hidden,int n_out);
void freeBPNN(SBPNN *);
bool test(SBPNN *, double *input_unit,int input_num,double *target,int target_num);

void train(SBPNN *, double *input_unit,int input_num, double *target,int target_num, double *eo, double *eh);

void adjustWeights(double *delta, int ndelta, double *ly, int nly, double** w, double **oldw, double eta, double momentum);
void getHiddenError(double* delta_h, int nh, double *delta_o, int no, double **who, double *hidden, double *err);
void getOutputError(double *delta, double *target, double *output, int nj, double *err);
void layerforward(double *l1, double *l2, double **conn, int n1, int n2);
double sigmoidal(double x);
#endif /* SBPNN_hpp */
