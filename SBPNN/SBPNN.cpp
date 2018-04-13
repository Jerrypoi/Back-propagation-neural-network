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
    
//    SBPNN() {
//        input_units = new double[input_n];
//        hidden_units = new double[hidden_n];
//        output_units = new double[output_n];
//        hidden_delta = new double[hidden_n];
//        output_delta = new double[output_n];
//        input_weights = new double*[input_n + 1];
//        for(int i = 0;i < input_n + 1;i++) {
//            input_weights[i] = new double[hidden_n];
//        }
//
//        //TODO: Initilize the weights here
//
//        hidden_weights = new double*[hidden_n + 1];
//        for(int i = 0;i < input_n;i++) {
//            hidden_weights[i] = new double[output_n];
//        }
//        //TODO: initilize input_prev_weights and hidden..
//        eta = 0.3;
//        momentum = 0.3;
//    }
};

void initSeed(int seed);     // 设置随机数生成器种子
void readBPNN(SBPNN *, char* filename);    // 读取保存的网络配置及加权系数
void saveBPNN(SBPNN *, char *filename);    // 保存当前的网络配置及加权系数
SBPNN* createBPNN(int n_in,int n_hidden,int n_out);  //创建一个BP网络，并初始化权值(随机化intput_weidhts和hidden_weights在-0.05到0.05之间, 清零input_prev_weights和hidden_prev_weights)
void freeBPNN(SBPNN *);
void test(SBPNN *, double *input_unit,int input_num,double *target,int target_num);  //测试, 给定输入input_unit, 返回前向传播后得到的输出target
//TODO: Implement test();

void train(SBPNN *net, double *input_unit,int input_num, double *target,int target_num, double *eo, double *eh); //训练样本中的某个输入向量input_unit和期望输出向量target,eo为本次训练输出层误差,eh为隐含层误差// eh 怎么求？

void adjustWeights(double *delta, int ndelta, double *ly, int nly, double** w, double **oldw, double eta, double momentum); //更新加权系数,delta是隐藏层或输入层的反向误差, ly是隐藏层或输入层的输出, w是隐藏层或输入层的加权系数, oldw是隐藏层或输入层上一次更新的加权系数

void getHiddenError(double* delta_h, int nh, double *delta_o, int no, double **who, double *hidden, double *err);//计算反向调节时的隐藏层误差delta_h,delta_o是输出层误差,who是隐藏层到输出层的加权系数, hidden是隐藏层的实际输出,err是隐藏层各节点误差绝对值的总和

void getOutputError(double *delta, double *target, double *output, int nj, double *err);  //计算反向调节时的输出层误差delta,target是输出层期望输出,output是输出层实际输出,err是输出层各节点误差绝对值的总和

void layerforward(double *l1, double *l2, double **conn, int n1, int n2); //执行一次l1到l2的前向传播, conn是连接的加权系数
double sigmoidal(double x);


//3. 关于BP神经网络的训练算法train的说明:
//对于训练样本中的某个样本:
//{
//    layerforward(...);//从输入到隐藏前向传播
//    layerforward(...);//从输入到隐藏前向传播
//
//    getOutputError(...);   //计算输出层误差
//    getHiddenError(...);   //计算隐藏层误差
//
//    adjustWeights(...); //调整隐藏层到输出层加权系数
//    adjustWeights(...); //调整输入层到隐藏层加权系数
//}
void train(SBPNN *net, double *input_unit,int input_num, double *target,int target_num, double *eo, double *eh) {
    for(int i = 0;i < input_num;i++) {
        net->input_units[i] = input_unit[i];
    }
    for(int i = 0;i < target_num;i++) {
        net->target[i] = target[i];
    }
    layerforward(net->input_units, net->hidden_units, net->hidden_weights, net->input_n, net->hidden_n);
    layerforward(net->hidden_units, net->output_units, net->hidden_weights, net->hidden_n, net->output_n);
    getOutputError(net->output_delta, net->target, net->output_units, net->output_n, eo);
    getHiddenError(net->hidden_delta, net->hidden_n, net->output_delta, net->output_n, net->hidden_weights, net->hidden_units, eh);
    adjustWeights(net->output_delta, net->output_n, net->hidden_units, net->hidden_n, net->hidden_weights, net->hidden_prev_weights, net->eta, net->momentum);
    adjustWeights(net->hidden_delta, net->hidden_n, net->input_units, net->input_n, net->input_weights, net->input_prev_weights, net->momentum, net->eta);
    
    
}

void adjustWeights(double *delta, int ndelta, double *ly, int nly, double** w, double **oldw, double eta, double momentum) {
//    hidden_weights = hidden_prev_weights + eta*output_delta*hidden_units + momentum*hidden_prev_weights
    for(int i = 0;i < nly;i++) {
        for(int j = 0;j < ndelta;j++) {
            if(i == 0) {
                double temp = w[i][j];
                w[i][j] = oldw[i][j] + eta * delta[j] * 1 + momentum * oldw[i][j];
                oldw[i][j] = temp;
            }
            else {
                double temp = w[i][j];
                w[i][j] = oldw[i][j] + eta * delta[j] * ly[i] + momentum * oldw[i][j];
                oldw[i][j] = temp;
            }
        }
    }
}

double sigmoidal(double x) {
    return 1 / (1 + exp(-x));
}

void layerforward(double *l1, double *l2, double **conn, int n1, int n2) {
    for(int i = 0;i < n2;i++) {
        l2[i] = conn[0][i];
        for(int j = 1;j <n1 + 1;j++)
            l2[i] += l1[j] * conn[j][i];
        l2[i] = sigmoidal(l2[i]);
    }
}

void getOutputError(double *delta, double *target, double *output, int nj, double *err) {
    *err = 0;
    for(int i = 0;i < nj;i++) {
        delta[i] = -(target[i] - output[i])*(output[i] - (1 - output[i]));
        *err += abs(target[i] - output[i]);
    }
}


void getHiddenError(double* delta_h, int nh, double *delta_o, int no, double **who, double *hidden, double *err) {
    //nh = inputweights[?][nh]
    
    *err = 0;

    for(int i = 1; i < nh;i++) {
        double sum = 0;
        for(int j = 0;j < no;j++)
            sum += delta_o[j] * who[i + 1][j];
        delta_h[i] = sum * hidden[i] * (1 - hidden[i]);
// ?        err +=
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
    if(file.is_open()) {
        file<<net->input_n<<endl;
        file<<net->hidden_n<<endl;
        file<<net->output_n<<endl;
        for(int i = 0;i < net->input_n + 1;i++)
            for(int j = 0;j < net->hidden_n;j++) {
                file<<net->input_weights[i][j]<<endl;
            }
        
        for(int i = 0;i < net->hidden_n + 1;i++)
            for(int j = 0;j < net->output_n;j++) {
                file<<net->hidden_weights[i][j]<<endl;
            }
    }
    file.close();
}

void readBPNN(SBPNN * net, char* filename) {
    std::ifstream file(filename);
    if(file.is_open()) {
        int intput_num,hidden_num,output_num;
        file>>intput_num;
        file>>hidden_num;
        file>>output_num;
        net = createBPNN(intput_num, hidden_num, output_num);
        for(int i = 0;i < net->input_n + 1;i++)
            for(int j = 0;j < net->hidden_n;j++) {
                file>>net->input_weights[i][j];
            }
        
        for(int i = 0;i < net->hidden_n + 1;i++)
            for(int j = 0;j < net->output_n;j++) {
                file>>net->hidden_weights[i][j];
            }
        file.close();
    }
    else {
        std::cerr<<"Open file failure!"<<endl;
    }

}


void initSeed(int seed = (int)time(NULL)) {
    srand(seed);
}

double fRand(double fMin, double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}


SBPNN * createBPNN(int n_in,int n_hidden,int n_out) {
    SBPNN * bpnn = new SBPNN;
    bpnn->input_n = n_in;
    bpnn->hidden_n = n_hidden;
    bpnn->output_n = n_out;
    bpnn->input_units = new double[bpnn->input_n];
    bpnn->hidden_units = new double[bpnn->hidden_n];
    bpnn->output_units = new double[bpnn->output_n];
    bpnn->target = new double[bpnn->output_n];
    bpnn->hidden_delta = new double[bpnn->hidden_n];
    bpnn->output_delta = new double[bpnn->output_n];
    
    bpnn->input_weights = new double*[bpnn->input_n + 1];
    for(int i = 0;i < bpnn->input_n + 1;i++) {
        bpnn->input_weights[i] = new double[bpnn->hidden_n];
        for(int j = 0;j < bpnn->hidden_n;j++) {
            bpnn->input_weights[i][j] = fRand(-0.05, 0.05);
        }
    }
    
    bpnn->hidden_weights = new double*[bpnn->hidden_n + 1];
    
    for(int i = 0;i < bpnn->hidden_n + 1;i++) {
        bpnn->hidden_weights[i] = new double[bpnn->output_n];
        for(int j = 0;j < bpnn->output_n;j++) {
            bpnn->hidden_weights[i][j] = fRand(-0.05, 0.05);
        }
    }
    
    bpnn->input_prev_weights = new double*[bpnn->input_n + 1];
    for(int i = 0;i < bpnn->input_n + 1;i++) {
        bpnn->input_prev_weights[i] = new double[bpnn->hidden_n];
        for(int j = 0;i < bpnn->hidden_n;j++) {
            bpnn->input_prev_weights[i][j] = 0;
        }
    }
    bpnn->hidden_prev_weights = new double*[bpnn->hidden_n + 1];
    for(int i = 0;i < bpnn->hidden_n + 1;i++) {
        bpnn->hidden_prev_weights[i] = new double[bpnn->output_n];
        for(int j = 0;j < bpnn->output_n;j++) {
            bpnn->hidden_prev_weights[i][j] = 0;
        }
    }
    bpnn->eta = 0.3;
    bpnn->momentum = 0.3;
    return bpnn;
}
