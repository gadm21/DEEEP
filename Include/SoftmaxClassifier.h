

#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "OutputLayer.h" 

class SoftmaxClassifier : public OutputLayer{
private:
    Tensor<double> output_;
    Tensor<double> apply(Tensor<double> & input) override;
public:
    Tensor<double> predict(Tensor<double> & input) override;
    pair<double, Tensor<double> > backprop(Tensor<int> ground_truth) override;

    double crossEntropy(Tensor<double> & y_hat, vector<int> & y);
    Tensor<double> crossEntropyPrime(Tensor<double> & output, vector<int> & y);
};

#endif 
