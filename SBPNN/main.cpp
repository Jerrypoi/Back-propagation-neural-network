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
    double input[input_n];
    double out[out_n + 1];
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
            for(int i = 0;i < input_n;i++) {
                file>>input[i];
            }
            file>>res;
            for(int i = 0;i <= out_n;i++) {
                out[i] = 0;
            }
            out[res + 1] = 1.0;
            correct = test(net, input, input_n, out, out_n);

            if(correct == false) {
                wrong_case++;
            }
            else {
                cout<<"Testing sample succeeded!!!"<<endl;
            }
        }
        cout<<"testing failure rate="<<(double)100 * wrong_case/all_test_case<<"%"<<endl;
        freeBPNN(net);
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
        train_times++;
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
        cout<<"No. "<<train_times<<"Train completed, err = "<<err<<endl;
        file.close();
        file.open(filename);
    }
    strcpy(filename, "weights");
    saveBPNN(net, filename);
    freeBPNN(net);
    return 0;
}
