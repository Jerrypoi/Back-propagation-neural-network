//
//  CBPNet.cpp
//  SBPNN
//
//  Created by Jerry Zhang on 2018/4/25.
//  Copyright © 2018 Jerry Zhang. All rights reserved.
//

#include "CBPNet.hpp"
BPNet::BPNet(int n_in,int n_hidden,int n_out) {
    initSeed();
    input_n = n_in;
    hidden_n = n_hidden;
    output_n = n_out;
    input_units = new double[input_n];
    hidden_units = new double[hidden_n];
    output_units = new double[output_n];
    target = new double[output_n];
    hidden_delta = new double[hidden_n];
    output_delta = new double[output_n];
    input_weights = new double*[input_n + 1];
    
    for(int i = 0;i < input_n + 1;i++) {
        input_weights[i] = new double[hidden_n];
        for(int j = 0;j < hidden_n;j++) {
            input_weights[i][j] = fRand(-0.05, 0.05);
        }
    }
    
    hidden_weights = new double*[hidden_n + 1];
    
    for(int i = 0;i < hidden_n + 1;i++) {
        hidden_weights[i] = new double[output_n];
        for(int j = 0;j < output_n;j++) {
            hidden_weights[i][j] = fRand(-0.05, 0.05);
        }
    }
    
    input_prev_weights = new double*[input_n + 1];
    for(int i = 0;i < input_n + 1;i++) {
        input_prev_weights[i] = new double[hidden_n];
        for(int j = 0;j < hidden_n;j++) {
            input_prev_weights[i][j] = input_weights[i][j];
        }
    }
    hidden_prev_weights = new double*[hidden_n + 1];
    for(int i = 0;i < hidden_n + 1;i++) {
        hidden_prev_weights[i] = new double[output_n];
        for(int j = 0;j < output_n;j++) {
            hidden_prev_weights[i][j] = hidden_weights[i][j];
        }
    }
    eta = 0.1;
    momentum = 0.3;
    std::cout<<"Create SBNPP success."<<std::endl;
}
BPNet::BPNet(std::ifstream &file) {
    if(file.is_open()) {
        int intput_num,hidden_num,output_num;
        file>>intput_num;
        file>>hidden_num;
        file>>output_num;
        input_n = intput_num;
        hidden_n = hidden_num;
        output_n = output_num;
        
        input_units = new double[input_n];
        hidden_units = new double[hidden_n];
        output_units = new double[output_n];
        target = new double[output_n];
        hidden_delta = new double[hidden_n];
        output_delta = new double[output_n];
        input_weights = new double*[input_n + 1];
        
        for(int i = 0;i < input_n + 1;i++) {
            input_weights[i] = new double[hidden_n];
            for(int j = 0;j < hidden_n;j++) {
                file>>input_weights[i][j];
            }
        }
        
        input_prev_weights = new double*[input_n + 1];
        for(int i = 0;i < hidden_n + 1;i++) {
            hidden_weights[i] = new double[output_n];
            for(int j = 0;j < output_n;j++) {
                file>>hidden_weights[i][j];
            }
        }
        input_prev_weights = new double*[input_n + 1];
        for(int i = 0;i < input_n + 1;i++) {
            input_prev_weights[i] = new double[hidden_n];
            for(int j = 0;j < hidden_n;j++) {
                input_prev_weights[i][j] = input_weights[i][j];
            }
        }
        hidden_prev_weights = new double*[hidden_n + 1];
        for(int i = 0;i < hidden_n + 1;i++) {
            hidden_prev_weights[i] = new double[output_n];
            for(int j = 0;j < output_n;j++) {
                hidden_prev_weights[i][j] = hidden_weights[i][j];
            }
        }
        eta = 0.1;
        momentum = 0.3;
    }
    else {
        std::cerr<<"Open file failure!"<<std::endl;
    }

}
void BPNet::saveBPNN(std::ofstream &file) {
    if(file.is_open()) {
        file<<input_n<<std::endl;
        file<<hidden_n<<std::endl;
        file<<output_n<<std::endl;
        for(int i = 0;i < input_n + 1;i++)
            for(int j = 0;j < hidden_n;j++) {
                file<<input_weights[i][j]<<std::endl;
            }
        
        for(int i = 0;i < hidden_n + 1;i++)
            for(int j = 0;j < output_n;j++) {
                file<<hidden_weights[i][j]<<std::endl;
            }
        
    }
    else {
        std::cerr<<"Open file failure!"<<std::endl;
    }
}
BPNet::~BPNet() {
    delete [] input_units;
    delete [] hidden_units;
    delete [] output_units;
    delete [] target;
    delete [] hidden_delta;
    delete [] output_delta;
    for(int i = 0;i < input_n + 1;i++) {
        delete [] input_weights[i];
    }
    delete [] input_weights;
    
    for(int i = 0;i < hidden_n + 1;i++) {
        delete [] hidden_weights[i];
    }
    delete [] hidden_weights;
    
    
    for(int i = 0;i < input_n + 1;i++) {
        delete [] input_prev_weights[i];
    }
    delete [] input_prev_weights;
    
    for(int i = 0;i < hidden_n + 1;i++) {
        delete [] hidden_prev_weights[i];
    }
    delete [] hidden_prev_weights;
}
void BPNet::layerforward() {
    for(int i = 0;i < hidden_n;i++) {
        hidden_units[i] = input_weights[0][i];
        for(int j = 1;j < input_n + 1;j++) {
            hidden_units[i] += input_units[j - 1] * input_weights[j][i];
        }
        hidden_units[i] = sigmoidal(hidden_units[i]);
    }
    for(int i = 0;i < output_n;i++) {
        output_units[i] = hidden_weights[0][i];
        for(int j = 1;j < hidden_n + 1;j++ ) {
            output_units[i] += hidden_units[j - 1] * hidden_weights[j][i];
        }
        output_units[i] = sigmoidal(output_units[i]);
    }
}
double BPNet::getError() {
    double err = 0;
    for(int i = 0;i < output_n;i++) {
        output_delta[i] = -(target[i] - output_units[i])*output_units[i] * (1 - output_units[i]);
        err += fabs(target[i] - output_units[i]);
    }
    for(int i = 0; i < hidden_n;i++) {
        double sum = 0;
        for(int j = 0;j < output_n;j++)
            sum += output_delta[j] * hidden_weights[i + 1][j];
        hidden_delta[i] = sum * hidden_units[i] * (1 - hidden_units[i]);
    }
    return err;
}

void BPNet::adjustWeights() {
    for(int i = 0;i < hidden_n + 1;i++) {
        for(int j = 0;j < output_n;j++) {
            if(i == 0) {
                double temp = hidden_weights[i][j];
                //w[i][j] = oldw[i][j] + eta * delta[j] * 1 + momentum * oldw[i][j];
                hidden_weights[i][j] -= eta * output_delta[j] * 1;
                hidden_prev_weights[i][j] = temp;
            }
            else {
                double temp = hidden_weights[i][j];
                //w[i][j] = oldw[i][j] + eta * delta[j] * ly[i] + momentum * oldw[i][j];
                hidden_weights[i][j] = hidden_weights[i][j] - eta * output_delta[j] * output_units[j];
                hidden_prev_weights[i][j] = temp;
            }
        }
    }
    
    for(int i = 0;i < input_n + 1;i++) {
        for(int j = 0;j < hidden_n;j++) {
            if(i == 0) {
                double temp = input_weights[i][j];
                input_weights[i][j] -= eta * hidden_delta[j] * 1;
                input_prev_weights[i][j] = temp;
            }
            else {
                double temp = input_weights[i][j];
                input_weights[i][j] = input_weights[i][j] - eta * hidden_delta[j] * input_units[i - 1];
                input_prev_weights[i][j] = temp;
            }
        }
    }
}
double BPNet::train(std::ifstream &file) {
    double res;
    for(int i = 0;i < input_n;i++)
        file>>input_units[i];
    for(int i = 0;i < output_n;i++)
        file>>target[i];
    layerforward();
    res = getError();
    adjustWeights();
    return res;
}

