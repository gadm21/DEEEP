

#ifndef LRSCHEDULER_H
#define LRSCHEDULER_H


class LRScheduler{
public:
    double learning_rate;
    virtual void onIterationEnd(int iteration) = 0;
};


#endif
