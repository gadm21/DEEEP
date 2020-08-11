

#include <bits/stdc++.h>
using namespace std;


int main(int argc, char ** argv){
    if (argc < 2) throw runtime_error("provide data directory");
    string data_path = argv[1];

    MNISTDataLoader trian_loader(data_path);

    vector<Module * > modules = {
        new Conv2d(1,8,3,1,0),
        new MaxPool(2,2),
        new Relu(),
        new FullyConnected(1200, 30),
        new Relu(),
        new FullyConnected(30, 10)
    }
    auto lr_scheduler = new LinearLRScheduler(0.2, -0.000005);
    NetworkModel model = NetworkModel(modules, new SoftmaxClassifier(), lr_scheduler);

    int epochs = 10;
    while(epochs--){
        while(train_loader.batch()){
            pair<Tensor<double>, vector<int> > xy = train_loader.nextBatch();
            double loss = model.trainStep(xy.first, xy.second);
        }
    }

    model.save("network.deeep")
    

}