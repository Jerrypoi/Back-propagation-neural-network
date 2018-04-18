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
using namespace std;

int main(int argc, const char * argv[]) {
    
    long long count = 0;
    const int input_n = 28 * 28;
    const int hidden_n = 64;
    const int out_n = 10;
    char do_what;
    double err = 100;
    double temp;
    int i;
    double input[input_n];
    double out[out_n];
    initSeed((int)time(NULL));
    SBPNN *net = createBPNN(input_n, hidden_n, out_n);
    char filename[100] = "sample_data";
    cout<<"Do you want to skip training network, redirect to testing it? Enter y for yes";
    cin>>do_what;
    if(do_what == 'y') {
        cout<<"Enter result file name:";
        cin>>filename;
        ifstream file(filename);
        while(!file.is_open()) {
            cout<<"Opening file failure, enter a new name:";
            cin>>filename;
            file.open(filename);
        }
        file.close();
        readBPNN(net, filename);
        cout<<"Enter testing sample name: ";
        cin>>filename;
        file.open(filename);
        while(!file.is_open()) {
            cout<<"Opening file failure, enter a new name:";
            cin>>filename;
            file.open(filename);
        }
        while(!file.eof()) {
            bool correct = true;
            double res;
            for(int i = 0;i < input_n;i++) {
                file>>net->input_units[i];
            }
            //TODO: rewrite codes here.
            file>>res;
            
            layerforward(net->input_units, net->hidden_units, net->input_weights, net->input_n, net->hidden_n);
            layerforward(net->hidden_units, net->output_units, net->hidden_weights, net->hidden_n, net->output_n);
            
            for(int i = 0;i < out_n;i++) {
                if(i != res) {
                    if(fabs(net->output_units[i] - 0) > 1e-4) {
                        correct = false;
                    }
                }
                else {
                    if(fabs(net->output_units[i] - 1) > 1e-4) {
                        correct = false;
                    }
                }

            }
            
            if(correct == false) {
                cout<<"testing failure!"<<endl;
                cout<<"Expecting "<<res<<" but got:"<<endl;
                for(int i = 0;i < out_n;i++) {
                    cout<<net->output_units[i]<<" ";
                }
                cout<<endl;
                cout<<endl;
                
            }
            else {
                cout<<"Testing sample succeeded!!!"<<endl;
            }
            
        }
        cout<<"Testing completed."<<endl;
        return 0;

    }
    
    ifstream file(filename);
    while(!file.is_open()) {
        cout<<"Opening file failure! Enter new file name"<<endl;
        cin>>filename;
        file.open(filename);
    }

    cout<<"file open success."<<endl;
    ofstream save_err_file("err");
    while(err > 1) {

        err = 0;
        double err_temp = 0;
        for(i = 0;i < input_n;i++)
            file>>input[i];
        for(i = 0;i < out_n;i++)
            file>>out[i];
        train(net, input, input_n, out, out_n, &err_temp, &temp);
        while (!file.eof()) {
            
            //outputing input_weights
//            for(i = 0;i < net->input_n + 1;i++) {
//                for(int j = 0;j < net->hidden_n;j++)
//                    cout<<net->input_weights[i][j]<<" ";
//                cout<<endl;
//            }
            
            for(i = 0;i < input_n;i++)
                file>>input[i];
            for(i = 0;i < out_n;i++)
                file>>out[i];
            train(net, input, input_n, out, out_n, &err_temp, &temp);
            err += err_temp;
//            cout<<"got input:"<<endl;
//            for(i = 0;i < 28;i++) {
//                for(int j = 0;j < 28;j++)
//                    cout<<net->input_units[i * 28 + j]<<" ";
//                cout<<endl;
//            }
//            cout<<"hidden_units"<<endl;
//            for(i = 0;i < hidden_n;i++)
//                cout<<net->hidden_units[i]<<" ";
//            cout<<endl;
//            cout<<"out_units"<<endl;
//            for(i = 0;i < out_n;i++)
//                cout<<net->output_units[i]<<" ";
            cout<<endl;
            cout<<"err_temp ="<<err_temp<<endl;
        }
        
        save_err_file<<err<<endl;
        count = 0;
        file.close();
        file.open(filename);
    }
    save_err_file.close();
    cout<<"err = "<<err<<endl;
    file.close();
//    //Use this for testing.
//    for(int i = 0;i < input_n + 1;i++) {
//        for(int j = 0;j < hidden_n;j++)
//            cout<<net->input_weights[i][j]<<" ";
//        cout<<endl;
//    }
    char save_filename[100] ="resultdata";
    saveBPNN(net, save_filename);
    
    cout<<"Train completed."<<endl;
    freeBPNN(net);
    
    return 0;
}
