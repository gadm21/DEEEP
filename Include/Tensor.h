

#ifndef TENSOR_H
#define TENOSR_H

#include <bits/stdc++.h>


/**
 * Tensor class _ supports from 1 to 4 dimensions
 * 
 * data format is in NCHW
 * */

template < typename T>
class Tensor{
private:
    T * data_ ;
    int size_ = -1;  // -1 means the size is undefined

    int num_dims = 0 ;
    int dims [4] {} ; 
public:
    Tensor () = default;
    Tensor (int num_dims, int const * dims) ;
    Tensor (const Tensor<T> & other);

    void view (int new_num_dims, int * new_dims) ;
    void zero () ;
    
    T get(int i) ;
    T get(int i, int j) ; 
    T get(int i, int j, int k) ;
    T get(int i, int j, int k, int l) ;

    void set(int i, T value) ;
    void set(int i, int j, T value) ;
    void set(int i, int j, int k, T value);
    void set(int i, int j, int k, int l, T value) ;

    void add(int i, T value);
    void add(int i, int j, int k, int l, T value) ;

    // matrix multiplication 
    Tensor<T> matmul(Tensor<T> other) ;
    Tensor<T> convolve2d(Tensor<T> kernels, int stride, int padding, Tensor<T> bias) ;
    Tensor<T> transpose() ;

    T sum();

    Tensor<T> operator+(Tensor<T> & other); //sum of two 2d tensors
    Tensor<T> operator*(Tensor<T> & other); //element-wise mutliplication of two 2d tensors
    Tensor<T> operator*(T mutliplier); //multiply every element of the tensor by a value  
    Tensor<T> operator/(T divisor); //divide every element of the tensor by a value
    Tensor<T> operator-=(Tensor<T> other) ;
    Tensor<T> & operator=(const Tensor<T> & other);

    Tensor<T> columnWiseSum();
    //Tensor<T> channelWiseSum();
    //initializes a tensor's value from a distribution 
    void randn(std::default_random_engine_generator, std::normal_distribution<double> distribution, double mutliplier);
    void print()



    virtual ~Tensor();

}