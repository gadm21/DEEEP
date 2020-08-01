


#include "../include/SoftmaxClassifier.h" 
#include "../include/Tensor.h" 


Tensor<double> SoftmaxClassifier::apply(Tensor<double> input){
    assert (input.num_dims==2);

    int rows = input.dims[0], cols = input.dims[1];
    Tensor<double> probabilities(2, input.dims);

    for (int row =0; row<rows; row++){
        double denominator = 0;
        for (int col=0; col<cols; col++) denominator += exp(input.get(row,col));
        for (int col=0; col<cols; col++) probabilities.set(row, col, input.get(i,j)/denominator);

    }
    
    return probabilities;
}

Tensor<double> SofmaxClassifier::predict(Tensor<double> input){
    output_ = apply(input);
    return output_;
}

pair<double, Tensor<double> > SoftmaxClassifier::backprop(Tensor<int> ground_truth){
    double loss = crossEntropy(output_, ground_truth);
    Tensor<double> gradient = crossEntropyPrime(output_, ground_truth);

    return make_pair(loss, gradient);
}

double crossEntropy(Tensor<double> & y_hat, vector<int> & y){
    double total = 0;
    for(int i=0; i<y.size(); i++) 
}