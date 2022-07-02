#include <iostream>
#include "neuralnetwork/matrix.hpp"
#include "neuralnetwork/nn.hpp"
#include "neuralnetwork/layers.hpp"
#include "neuralnetwork/costs.hpp"

using namespace std;

#define NUM_LAYERS  4
#define LAYER1      (size_t)  5, (size_t) 10, sigmoid, sigmoidDerivative
#define LAYER2      (size_t) 10, (size_t)  6, relu   , reluDerivative
#define LAYER3      (size_t)  6, (size_t)  4, softmax, softmaxDerivative


int main(){
    std::normal_distribution<float> dist(-1, 1);
    NeuralNetwork nn (NUM_LAYERS, {new FCLayer(LAYER1), new FCLayer(LAYER2), new FCLayer(LAYER3)});

    nn.printInfo();

    Matrix<float> target(4, 4);
    target.set(1, 0, 0);
    target.set(1, 1, 1);
    target.set(1, 2, 3);
    target.set(1, 3, 3);

    float error = nn.train(Matrix<float>(4, 5, dist), target, 0.2, crossEntropy, crossEntropyDerivative);
    cout << "Error: " << error << endl;

    return 0;
}
