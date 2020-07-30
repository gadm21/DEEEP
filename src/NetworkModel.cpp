

#include "../include/NetworkModel.h" 
#include "../include/LRScheduler.h" 
#include "../include/Tensor.h" 

using namespace std;

NetworkModel::NetworkModel(vector<Module * > & modules, OutputLayer * output_layer, LRScheduler * lr_scheduler){
    modules_ = modules ; 
    lr_scheduler_ = lr_scheduler;
    output_layer_ = output_layer;
}


double NetworkModel::trainStep(Tensor<double> & x, vector<int> & y){
    //forward
    Tensor<double> output = forward(x);

    //backprop
    pair<double, Tensor<double> > loss_and_cost_gradient = output_layer_->backprop(y);
    Tensor<double> chain_gradient = loss_and_cost_gradient.second;
    for(int i = (int) modules_.size() -1; i>=0; i--)
        chain_gradient = modules_[i]->backprop(chain_gradient, lr_scheduler_->learning_rate);
    iteration++;
    lr_scheduler_->onIterationEnd(iteration);

    return loss_and_gradient.first;
}


Tensor<double> NetworkModel::forward(Tensor<double> & x){
    for(auto & module : modules_) x = module->forward(x);
    return output_layer.predict(x);
}


vector<int> NetworkModel::predict(Tensor<double> & x){
    Tensor<double output = forward(x);
    vector<int> predictions;
    for(int i=0; i<output.dims[0]; i++){
        int argmax = -1;
        double max = -1;
        for(int j=0; j<output.dims[1]; j++){
            if (output.get(i,j) > max){
                max = output.get(i,j);
                argmax = j;
            }
        }
        predictions.push_back(argmax);
    }
    return predictions;
}


void NetworkModel::load(string path){
    FILE * model_file = fopen(path.c_str(), "r");
    if(!model_file) throw runtime_error("error reading model file");
    for (auto & module : modules_) module->load(model_file);
}

void NetworkModel::save(string path){
    FILE * model_file = fopen(path.c_str(), "w");
    if(!model_file) throw runtime_error("error reading model file");
    for (auto & module : modules_) module->save(model_file);
}

NetworkModel::~NetworkModel(){
    for ( auto & module : modules_) delete module;
    delete output_layer_;
    delete lr_scheduler_;
}

void NetworkModel::eval(){
    for(auto & module : modules_) module->eval();
}