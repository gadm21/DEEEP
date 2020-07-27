

#ifndef TENSOR_H
#define TENOSR_H

#include <bits/stdc++.h>


/**
 * Tensor class _ supports from 1 to 4 dimensions
 * 
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

    //2D convolution 
    Tensor<T> convolve2d(Tensor<T> kernels, int stride, int padding, Tensor<T> bias) ;

    //Transpose
    Tensor<T> transpose() ;

    T sum();

    Tensor<T> operator+(Tensor<T> & other); //sum of two 2d tensors
    Tensor<T> operator*(Tensor<T> & other); //element-wise mutliplication of two 2d tensors
    Tensor<T> operator*(T mutliplier); //multiply every element of the tensor by a value  
    Tensor<T> operator/(T divisor); //divide every element of the tensor by a value
    

}