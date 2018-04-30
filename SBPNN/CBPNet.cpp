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
    input_units = new double[input_n + 1];
    input_units[0] = 1.0;
    hidden_units = new double[hidden_n + 1];
    hidden_units[0] = 1.0;
    output_units = new double[output_n + 1];
    target = new double[output_n + 1];
    hidden_delta = new double[hidden_n + 1];
    output_delta = new double[output_n + 1];
    
    input_weights = new double*[input_n + 1];
    for(int i = 0;i < input_n + 1;i++) {
        input_weights[i] = new double[hidden_n + 1];
        for(int j = 0;j <= hidden_n;j++) {
            input_weights[i][j] = fRand(-0.05, 0.05);
        }
    }
    
    hidden_weights = new double*[hidden_n + 1];
    for(int i = 0;i < hidden_n + 1;i++) {
        hidden_weights[i] = new double[output_n + 1];
        for(int j = 0;j <= output_n;j++) {
            hidden_weights[i][j] = fRand(-0.05, 0.05);
        }
    }
    
    input_prev_weights = new double*[input_n + 1];
    for(int i = 0;i < input_n + 1;i++) {
        input_prev_weights[i] = new double[hidden_n + 1];
        for(int j = 0;j <= hidden_n;j++) {
            input_prev_weights[i][j] = 0;
        }
    }
    hidden_prev_weights = new double*[hidden_n + 1];
    for(int i = 0;i < hidden_n + 1;i++) {
        hidden_prev_weights[i] = new double[output_n + 1];
        for(int j = 0;j <= output_n;j++) {
            hidden_prev_weights[i][j] = 0;
        }
    }
    eta = 0.5;
    momentum = 0.5;
    std::cout<<"Create CBNPP success."<<std::endl;
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
        input_units = new double[input_n + 1];
        input_units[0] = 1.0;
        hidden_units = new double[hidden_n + 1];
        hidden_units[0] = 1.0;
        output_units = new double[output_n + 1];
        target = new double[output_n + 1];
        hidden_delta = new double[hidden_n + 1];
        output_delta = new double[output_n + 1];
        
        input_weights = new double*[input_n + 1];
        for(int i = 0;i < input_n + 1;i++) {
            input_weights[i] = new double[hidden_n + 1];
            for(int j = 0;j <= hidden_n;j++) {
                input_weights[i][j] = fRand(-0.05, 0.05);
            }
        }
        
        hidden_weights = new double*[hidden_n + 1];
        for(int i = 0;i < hidden_n + 1;i++) {
            hidden_weights[i] = new double[output_n + 1];
            for(int j = 0;j <= output_n;j++) {
                hidden_weights[i][j] = fRand(-0.05, 0.05);
            }
        }
        
        input_prev_weights = new double*[input_n + 1];
        for(int i = 0;i < input_n + 1;i++) {
            input_prev_weights[i] = new double[hidden_n + 1];
            for(int j = 0;j <= hidden_n;j++) {
                input_prev_weights[i][j] = 0;
            }
        }
        hidden_prev_weights = new double*[hidden_n + 1];
        for(int i = 0;i < hidden_n + 1;i++) {
            hidden_prev_weights[i] = new double[output_n + 1];
            for(int j = 0;j <= output_n;j++) {
                hidden_prev_weights[i][j] = 0;
            }
        }
        eta = 0.3;
        momentum = 0.3;
        
        for(int i = 0;i < input_n + 1;i++)
            for(int j = 0;j <= hidden_n;j++) {
                file>>input_weights[i][j];
            }
        for(int i = 0;i < hidden_n + 1;i++)
            for(int j = 0;j <= output_n;j++) {
                file>>hidden_weights[i][j];
            }
        std::cout<<"Create CBNPP success."<<std::endl;
    }
}
void BPNet::saveBPNN(std::ofstream &file) {
    if(file.is_open()) {
        file<<input_n<<std::endl;
        file<<hidden_n<<std::endl;
        file<<output_n<<std::endl;
        for(int i = 0;i < input_n + 1;i++)
            for(int j = 0;j <= hidden_n;j++) {
                file<<input_weights[i][j]<<std::endl;
            }
        
        for(int i = 0;i < hidden_n + 1;i++)
            for(int j = 0;j <= output_n;j++) {
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
    for(int i = 1;i <= hidden_n;i++) {
        hidden_units[i] = 0;
        for(int j = 0;j <= input_n;j++) {
            hidden_units[i] += input_units[j] * input_weights[j][i];
        }
        hidden_units[i] = sigmoidal(hidden_units[i]);
    }
    for(int i = 1;i <= output_n;i++) {
        output_units[i] = 0;
        for(int j = 0;j <= hidden_n;j++ ) {
            output_units[i] += hidden_units[j] * hidden_weights[j][i];
        }
        output_units[i] = sigmoidal(output_units[i]);
    }
}
double BPNet::getError() {
    double err = 0;
    output_delta[0] = 0;
    for(int i = 1;i <= output_n;i++) {
        output_delta[i] = -(target[i] - output_units[i])*output_units[i] * (1 - output_units[i]);
        err += fabs(target[i] - output_units[i]);

    }
    for(int i = 0; i <= hidden_n;i++) {
        double sum = 0;
        for(int j = 0;j <= output_n;j++)
            sum += output_delta[j] * hidden_weights[i][j];
        hidden_delta[i] = sum * hidden_units[i] * (1 - hidden_units[i]);
    }
    return err;
}

void BPNet::adjustWeights() {
    for(int i = 0;i < hidden_n + 1;i++) {
        for(int j = 0;j <= output_n;j++) {
            //w[i][j] = oldw[i][j] + eta * delta[j] * ly[i] + momentum * oldw[i][j];
            hidden_weights[i][j] = hidden_weights[i][j] - eta * output_delta[j] * hidden_units[i] + momentum * hidden_prev_weights[i][j];
            hidden_prev_weights[i][j] = - eta * output_delta[j] * hidden_units[i] + momentum * hidden_prev_weights[i][j];
        }
    }
    for(int i = 0;i <= input_n;i++) {
        for(int j = 0;j <= hidden_n;j++) {
            input_weights[i][j] = input_weights[i][j] - eta * hidden_delta[j] * input_units[i] + momentum * input_prev_weights[i][j];
            input_prev_weights[i][j] = - eta * hidden_delta[j] * input_units[i] + momentum * input_prev_weights[i][j];
        }
    }
}
double BPNet::train(std::ifstream &file) {
    double err = 0;
    if(!file.eof()) {
        for(int i = 1;i <= input_n;i++) {
            file>>input_units[i];
        }
        for(int i = 1;i <= output_n;i++)
            file>>target[i];
        layerforward();
        err = getError();
        adjustWeights();
    }
//    std::cout<<"Error ="<<err<<std::endl;
    return err;
}
void BPNet::test(std::ifstream &file) {
    int all_test_case = 0;
    int wrong_case = 0;

    int result;
    while(!file.eof()) {
        bool is_correct = true;
        all_test_case++;
        for(int i = 1;i <= input_n;i++) {
            file>>input_units[i];
        }
        file>>result;
        layerforward();
        for(int i = 1;i <= output_n;i++) {
            if(i == result + 1) {
                if(fabs(output_units[i] - 1) > 0.3) {
                    wrong_case++;
                    is_correct = false;
                    break;
                }
            }
            else {
                if(fabs(output_units[i] - 0) > 0.3) {
                    wrong_case++;
                    is_correct = false;
                    break;
                }
            }
        }
        if(is_correct == false) {
            std::cout<<"Testing failure, expecting "<<result<<std::endl;
            std::cout<<"But got:"<<std::endl;
            for(int i = 1 ;i <= output_n;i++) {
                std::cout<<output_units[i]<<" ";
            }
            std::cout<<std::endl;
        }
        else
            std::cout<<"Testing success"<<std::endl;
    }
    std::cout<<"Testing failure rate = "<<(double)wrong_case/all_test_case * 100<<"%"<<std::endl;
}

