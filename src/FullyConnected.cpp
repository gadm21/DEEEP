

#include "../include/FullyConnected.h"
#include "../include/Tensor.h" 



FullyConnected::FullyConnected(int input_size, int output_size, int seed){
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);

    int weights_dims[] = {input_size, output_size};
    weights = Tensor<double>(2, weights_dims);
    weights.randn(generator, distribution, sqrt(2.0/input_size)) ;
    int bias_dims[] = {output_size};
    bias = Tensor<double> (1, bias_dims);
    bias.randn(generator, distribution, 0) ;
}

Tensor<double> & FullyConnected::forward(Tensor<double> & input){
    input_num_dims = input.num_dims;
    std::copy(input.dims, input.dims + input.num_dims, this->input_dims);

    if(input.num_dims != 2){
        int flatten_size = 1;
        for(int i=0; i<input.num_dims; i++) flatten *= input.dims[i];
        int new_dims[] = {input.dims[0], flatten_size};
        input.view(2, new_dims);
    }
    input_ = input;
    product_ = input.matmul(weights) + bias ; 
    return product_;
}

Tensor<double> & FullyConnected::backprop(Tensor<double> & chainGradient, double learning_rate){
    Tensor<double> weightGradient = input_.transpose().matmul(chainGradient); //derivative w.r.t weights
    Tensor<double> biasGradient = chainGradient.columnWiseSum(); //derivative w.r.t bias
    
    chainGradient = chainGradient.matmul(weights.transpose()); //derivative w.r.t inputs
    chainGradient.view(input_num_dims, input_dims) ;

    weights -= (learning_rate * weightGradient);
    bias -= (learning_rate * biasGradient);

    return chainGradient;
}

void FullyConnected::load(FILE * file_model){
    double value;
    for(int i=0; i<weights.dims[0]; i++)
        for(int j=0; j<weights.dims[1]; j++){
            int read = fscanf(file_model, "%lf", &value);
            if (read!=1) throw std::runtime_error("invalid model file");
            weights.set(i,j, value);
        }
    
    for(int i=0; i<bias.dims[0]; i++){
        int read = fscanf(file_model, "%lf", %value);
        if (read!=1) throw std::runtime_error("invalid model file");
        bias.set(i, value);
    }
}

void FullyConnected::save(FILE * file_model){
    for (int i=0; i<weights.dims[0]; i++)
        for(int j=0; j<weights.dims[1]; j++)
            fprintf(file_model, "%0.18f", weights.get(i,j));
    
    for(int i=0; i<bias.dims[0]; i++) fprintf(file_model, "%0.18f", bias.get(i));
}

