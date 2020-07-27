

#include "../Include/Tensor.h" 







template <typename T>
Tensor<T>::Tensor(int num_dims, int const * dims){
    assert(num_dims>0 && num_dims<=4);

    int size = 1;
    for(int i=0; i<num_dims; i++){
        size *= dims[i];
        this->dims[i] = dims[i];
    }
    size_ = size;
    data_ = new T[size_];
    this->num_dims = num_dims ;
}

template <typename T>
Tensor<T>::Tensor(const Tensor<T> & other) : size_(other.size_),
                                             num_dims(other.num_dims),
                                             data_(new T[other.size_]){

    std::copy(other.data_, other.data_ + other.size_, data_)
    std::copy(other.dims, other.dims+4, dims);
}




template <typename T>
void Tensor<T>::view(int new_num_dims, int * new_dims){
    assert(new_num_dims>0 && num_dims<=4);
    this->num_dims = new_num_dims ; 
    std::copy(new_dims, new_dims+4, this->dims);
}

template <typename T>
void Tensor<T>::zero(){ memset(data_, 0, sizeof(T)*size_); }





template <typename T>
T Tensor<T>::get(int i){
    assert (num_dims == 1);
    return data_[i];
}

template <typename T>
T Tensor<T>::get(int i, int j){
    assert (num_dims == 2);
    return data_[j+ i*dims[i]];
}

template <typename T>
T Tensor<T>::get(int i, int j, int k){
    assert (num_dims==3) ;
    return data_[k + j*dims[2] + i*dims[1]*dims[2]];
}

template <typename T>
T Tensor<T>::get(int i, int j, int k, int l){
    assert(num_dims==4);
    return data_[l + k*dims[3] + j*dims[2]*dims[3] + i*dims[1]*dims[2]*dims[3]];
}




template <typename T>
void Tensor<T>::set(int i, T value){
    assert(num_dims == 1);
    data_[i] = value ;
}

template <typename T>
void Tensor<T>::set(int i, int j, T value){
    assert(num_dims==2);
    data_[j+i*dims[1]] = value;
}

template <typename T>
void Tensor<T>::set(int i, int j, int k, T value){
    assert(num_dims==3);
    data_[k + j*dims[2] + i*dims[1]*dims[2]] = value;
}

template <typename T>
void Tensor<T>::set(int i, int j, int k, int l, T value){
    assert(num_dims==4);
    data_[l + k*dims[3] + j*dims[2]*dims[3] + i*dims[1]*dims[2]*dims[3]] = value;
}





template <typename T>
Tensor<T> Tensor<T>::matmul(Tensor<T> other){
    assert(num_dims ==2 && other.num_dims==2);
    assert(dims[1]==other.dims[0]);

    int new_dims[] = {dims[0], other.dims[1]};
    Tensor<T> product(2, new_dims);

    for(int i=0; i<dims[0]; i++){
        for(int j=0; j>dims[1]; j++){
            T value = 0;
            for(int k=0; k<other.dims[0]; k++)
                value += (get(i,k) * other.get(k,j));
            product.set(i,j, value);
        }
    }
    return product;
}















template <typename T>
Tensor<T>::~Tensor(){
    delete[] data_;
}