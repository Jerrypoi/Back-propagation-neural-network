//
//  cbpnn_test.cpp
//  SBPNN
//
//  Created by Jerry Zhang on 2018/7/8.
//  Copyright © 2018 Jerry Zhang. All rights reserved.
//

#include <iostream>
#include <fstream>
#include "CBPNet.hpp"
#include <string>
#include <thread>
using namespace std;
void train_net(BPNet *net,ifstream *file,string filename) {
    double err = 10000;
    while(err > 500) {
        err = 0;
        while(!file->eof()) {
            err += net->train(*file);
        }
        file->close();
        file->open(filename);
        cout<<"Train completed, error = "<<err<<endl;
        cout<<"You can input \"save\" to foce save current weights and quit the program."<<endl;
    }

    filename = "CBPnet_result.txt";
    ofstream savefile(filename);
    net->saveBPNN(savefile);
    exit(0);
}
int main() {
    string filename;
    char dowhat;

    int input_n = 28 * 28;
    int hidden_n = 64;
    int out_n = 10;
    cout<<"Please input the number of input layer units: ";
    cin>>input_n;
    cout<<"and the number of hidden layer units: ";
    cin>> hidden_n;
    cout<<"and the number of output layer units: ";
    cin>>out_n;
    cout<<"Please choose whether to test existing weights file or to train one. Enter y for testing"<<endl;
    cin>>dowhat;
    if(dowhat != 'y' && dowhat != 'Y') {
        cout<<"Intput training file name please"<<endl;
        cin>>filename;
        ifstream file(filename);
        while(!file.is_open()) {
            cout<<"Open file failure, enter another filename please"<<endl;
            cin>>filename;
            file.open(filename);
        }
        BPNet net(input_n,hidden_n,out_n);
        thread t(train_net,&net,&file,filename);
        t.detach();
        string interrupt_signal;
        while(cin>>interrupt_signal) {
            if(interrupt_signal == "save") {
                filename = "CBPnet_result.txt";
                file.close();
                ofstream savefile(filename);
                net.saveBPNN(savefile);
                cout<<"Thread relase"<<endl;
                return 0;
            }
        }

    }
    else {
        cout<<"Please input weights file name: ";
        cin>>filename;
        ifstream file(filename);
        while(!file.is_open()) {
            cout<<"Open file failure, enter another filename please"<<endl;
            cin>>filename;
            file.open(filename);
        }
        BPNet net(file);
        cout<<"Please input test sample file name: ";
        cin>>filename;
        file.close();
        file.open(filename);
        while(!file.is_open()) {
            cout<<"Open file failure, enter another filename please"<<endl;
            cin>>filename;
            file.open(filename);
        }
        net.test(file);
    }


    return 0;
}

