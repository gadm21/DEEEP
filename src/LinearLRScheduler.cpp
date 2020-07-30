
#include "../include/LinearLRScheduler.h" 

LinearLRScheduler::LinearLRScheduler(double initial_lr, double step){
    learning_rate = initial_lr;
    step = step;
}

void LinearLRScheduler::onIterationEnd(int iteration){
    learning_rate += step;
}