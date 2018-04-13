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
struct SBPNN;
double fRand(double fMin, double fMax);
void initSeed(int seed);     // 设置随机数生成器种子
void readBPNN(SBPNN *, char* filename);    // 读取保存的网络配置及加权系数
void saveBPNN(SBPNN *, char *filename);    // 保存当前的网络配置及加权系数
SBPNN* createBPNN(int n_in,int n_hidden,int n_out);  //创建一个BP网络，并初始化权值(随机化intput_weidhts和hidden_weights在-0.05到0.05之间, 清零input_prev_weights和hidden_prev_weights)
void freeBPNN(SBPNN *);
void test(SBPNN *, double *input_unit,int input_num,double *target,int target_num);  //测试, 给定输入input_unit, 返回前向传播后得到的输出target
//TODO: Implement test();

void train(SBPNN *, double *input_unit,int input_num, double *target,int target_num, double *eo, double *eh); //训练样本中的某个输入向量input_unit和期望输出向量target,eo为本次训练输出层误差,eh为隐含层误差

void adjustWeights(double *delta, int ndelta, double *ly, int nly, double** w, double **oldw, double eta, double momentum); //更新加权系数,delta是隐藏层或输入层的反向误差, ly是隐藏层或输入层的输出, w是隐藏层或输入层的加权系数, oldw是隐藏层或输入层上一次更新的加权系数
void getHiddenError(double* delta_h, int nh, double *delta_o, int no, double **who, double *hidden, double *err);//计算反向调节时的隐藏层误差delta_h,delta_o是输出层误差,who是隐藏层到输出层的加权系数, hidden是隐藏层的实际输出,err是隐藏层各节点误差绝对值的总和
void getOutputError(double *delta, double *target, double *output, int nj, double *err);  //计算反向调节时的输出层误差delta,target是输出层期望输出,output是输出层实际输出,err是输出层各节点误差绝对值的总和
void layerforward(double *l1, double *l2, double **conn, int n1, int n2); //执行一次l1到l2的前向传播, conn是连接的加权系数
double sigmoidal(double x);
#endif /* SBPNN_hpp */
