

#ifndef LINEARLRSCHEDULER_H
#define LINEARLRSCHEDULER_H 

#include "LRScheduler.h" 

class LinearLRScheduler : public LRScheduler {
public:
    double step;
    LinearLRScheduler(double initial_lr, double step);
    void onIterationEnd(int iteration) override;
};

#endif 