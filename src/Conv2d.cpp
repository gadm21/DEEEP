

#include "../include/Conv2d.h" 


Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, int seed){
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);

    int kernel_dims = {out_channels, in_channels, kernel_size, kernel_size} 
    kernels = Tensor<double> (4, kernel_dims);
    kernels.randn(generator, distribution, sqrt(2.0/(kernel_size*kernel_size*out_channels)));

    int bias_dims[] = {out_channels} 
    bias = Tensor<double> (1, bias_dims);
    bias.randn(generator, distribution, 0);

    this->stride = stride;
    this->padding = padding;

}

Tensor<double> & Conv2d::forward(Tensor<double> & input){
    input_ = input;
    product_ = input.convolve2d(kernels, stride, padding, bais);

    return product_;
}

Tensor<double> Conv2d::backprop(Tensor<double> chain_gradient, double learning_rate){
    Tensor<double> kernels_gradient(kernels.num_dims, kernels.dims);
    Tensor<double> input_gradient(input_.num_dims, input_.dims);
    Tensor<double> bias_gradient(bias.num_dims, bias.dims);
    kernels_gradient.zero(); 
    input_gradient.zero(); 
    bias_gradient.zero();

    for(int batch=0; batch<input_.dims[0]; batch++){
        for(int filter=0; filter<kernels.dims[0]; filter++){
            int x= -padding;
            for(int cx=0; cx<chain_gradient.dims[2]; x += stride, cx++){
                int y = -padding;
                for(int cy=0; cy<chain_gradient.dims[3]; y+= stride, cy++){
                    double chain_grad = chain_gradient.get(batch, filter, cx, cy);
                    for(int fx=0; fx, kernels.dims[2]; fx++){
                        int ix = x + fx;
                        if (ix >=0 && ix<input_.dims[2]){
                            for(int fy=0; fy<kernels.dims[3]; fy++){
                                int iy = y + fy;
                                if (iy>=0 && iy<input_.dims[3]){
                                    for(int fc=0, fc<kernels.dims[1];fc++){
                                        kernels_gradient.add(filter, fc, fx, fy, input_.get(batch, fc, ix, iy)*chain_grad);
                                        input_gradient.add(batch, fc, ix, iy, kernels.get(filter, fc, fx, fy)*chain_grad);
                                    }
                                }
                            }
                        }
                    }
                    bias_gradient.add(filter, chain_grad)
                }
            }
        }
    }
    kernels -= kernels_gradient* learning_rate;
    bias -= bias_gradient * learning_rate;
    
    return input_gradient;
}