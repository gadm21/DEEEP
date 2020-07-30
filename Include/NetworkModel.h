
#ifndef NETWORKMODEL_H
#define NETWORKMODEL_H

#include "Tensor.h" 
#include "Module.h" 
#include "LRScheduler.h" 
using namespace std;

/**
 * train and test a neural network defined by Modules
 * */

class NetworkModel{
private:
    vector<Module * > modules_ ;
    OutputLayer * output_layer_ ;
    LRScheduler lr_scheduler_ ;
    int iteration = 0 ;
public:

    NetworkModel(vector<Module * > & modules, OutputLayer * output_layer, LRScheduler * lr_scheduler);

    double trainStep(Tensor<double> & x, vector<int> & y);
    Tensor<double> forward(Tensor<double> &x);
    vector<int> predict(Tensor<double> & x);
    void load(string path);
    void save(string path);

    virtual ~NetworkModel();
    void eval();

};

#endif