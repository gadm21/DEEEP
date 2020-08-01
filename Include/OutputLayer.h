

#ifndef OUTPUTLAYER_H
#define OUTPUTLAYER_H


#include "Tensor.h"


/**
 * Interface specific for model outputs
 * */
class OutputLayer{
protected:
    Tensor<double> apply(Tensor<double> input) = 0;
public:
    virtual Tensor<double> predict(Tensor<double> input) = 0;
    virtual pair<double, Tensor<double> > backprop(vector<int> ground_truth) = 0;
    virtual ~OutputLayer() = default();

};

#endif