//
//  main.cpp
//  SBPNN
//
//  Created by Jerry Zhang on 29/03/2018.
//  Copyright © 2018 Jerry Zhang. All rights reserved.
//
#include <iostream>
#include <string>
#include <ctime>

#include "SBPNN.hpp"
#include "CBPNet.hpp"

using namespace std;

int main(int argc, const char * argv[]) {
    long long train_times = 0;
    const int input_n = 28 * 28;
    const int hidden_n = 64;
    const int out_n = 10;
    char do_what;
    double err = 1000;
    int i;
    double input[input_n];
    double out[out_n];
    initSeed((int)time(NULL));
    SBPNN *net = createBPNN(input_n, hidden_n, out_n);
    char filename[100] = "train_sample.txt";
    cout<<"Do you want to skip training network, redirect to testing it? Enter y for yes";
    cin>>do_what;
    if(do_what == 'y') {
        long long all_test_case = 0;
        long long wrong_case = 0;
        cout<<"Enter result file name:";
        cin>>filename;
        ifstream file(filename);
        while(!file.is_open()) {
            cout<<"Opening file failure, enter a new name:";
            cin>>filename;
            file.open(filename);
        }
        file.close();
        net = readBPNN(net, filename);

        cout<<"Enter testing sample name: ";
        cin>>filename;
        file.open(filename);
        while(!file.is_open()) {
            cout<<"Opening file failure, enter a new name:";
            cin>>filename;
            file.open(filename);
        }
        while(!file.eof()) {
            all_test_case++;
            bool correct = true;
            int res;
            net->input_units[0] = 1.0;
            for(int i = 1;i <= input_n;i++) {
                file>>net->input_units[i];
            }
            file>>res;
            layerforward(net->input_units, net->hidden_units, net->input_weights, net->input_n, net->hidden_n);
            layerforward(net->hidden_units, net->output_units, net->hidden_weights, net->hidden_n, net->output_n);
            
            for(int i = 1;i <= out_n;i++) {
                if(i - 1 != res) {
                    if(fabs(net->output_units[i] - 0) > 0.3) {
                        correct = false;
                    }
                }
                else {
                    if(fabs(net->output_units[i] - 1) > 0.3) {
                        correct = false;
                    }
                }
            }
            if(correct == false) {
                wrong_case++;
                cout<<"testing failure!"<<endl;
                cout<<"Expecting "<<res<<" but got:"<<endl;
                for(int i = 1;i <= out_n;i++) {
                    cout<<net->output_units[i]<<" ";
                }
                cout<<endl;
                cout<<endl;
            }
            else {
                cout<<"Testing sample succeeded!!!"<<endl;
            }
        }
        cout<<"testing failure rate="<<(double)100 * wrong_case/all_test_case<<"%"<<endl;
        cout<<"Testing completed."<<endl;
        return 0;
    }
    ifstream file(filename);
    while(!file.is_open()) {
        cout<<"Opening file failure! Enter new file name for the training file."<<endl;
        cin>>filename;
        file.open(filename);
    }
    cout<<"file open success."<<endl;
    err = 10000;
    while(err > 500) {
        err = 0;
        double temp;
        double err_temp;
        while(!file.eof()) {
            for(int i = 0;i < input_n;i++) {
                file >> input[i];
            }
            for(int i = 0;i < out_n;i++)
                file>>out[i];
            train(net, input, input_n, out, out_n, &err_temp, &temp);
            
            err += err_temp;
        }
        if(err < 1000){
            for(int i = 1;i <= out_n;i++)
                cout<<net->output_units[i]<<" ";
            cout<<endl;}
        cout<<"Train completed, err = "<<err<<endl;
        file.close();
        file.open(filename);
        if(file.eof()) {
            cout<<"file reopen failure."<<endl;
        }
        
    }
    strcpy(filename, "weights");
    saveBPNN(net, filename);
    freeBPNN(net);
    /*
        Using these bunch of code to find why my err cant be minused after second train.
     */
    
//    BPNet bpnet(input_n,hidden_n,out_n);
//    err = 0;
//    while (!file.eof()) {
//        err += bpnet.train(file);
//    }
//    cout<<"After first train, err = "<<err<<endl;
//    file.close();
//    file.open(filename);
//    ofstream save_err_file("err");
//    while(err > 9400) {
//        err = 0;
//        while (!file.eof()) {
//            err += bpnet.train(file);
//        }
//        save_err_file<<err<<endl;
//        file.close();
//        file.open(filename);
//        cout<<"err = "<<err<<endl;
//        cout<<"trian times = "<<(++train_times)<<endl;
//    }
//    ofstream result_file("resultdata");
//    bpnet.saveBPNN(result_file);
//    save_err_file.close();
//    cout<<"err = "<<err<<endl;
//    file.close();
//
//    char save_filename[100] ="resultdata";
//    saveBPNN(net, save_filename);
//
//    cout<<"Train completed."<<endl;
//    freeBPNN(net);
    
    return 0;
}
