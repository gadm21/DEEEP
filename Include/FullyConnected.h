
#ifndef FULLYCONNECTED_H
#define FULLYCONNECTED_H 

#include "Module.h" 
#include "Tensor.h" 

class FullyConnected : public Module{
private:
    Tensor<double> weights ; 
    Tensor<double> bias ;
    Tensor<double> input_ ;
    Tensor<double> product_ ;

    int input_dims[4];
    int input_num_dims;
public: 
    FullyConnected(int input_size, int output_size, int seed = 0);
    
    Tensor<double> & forward(Tensor<double> & input) override;
    Tensor<double> & backprop(Tensor<double> chainGradient, double learning_rate) override;
    
    void load(FILE * file_model) override;
    void save(FILE * file_model) override;

};

#endif 