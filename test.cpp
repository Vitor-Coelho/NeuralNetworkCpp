#include <iostream>
#include "neuralnetwork/matrix.hpp"
#include "neuralnetwork/nn.hpp"
#include "neuralnetwork/layers.hpp"

using namespace std;

#define NUM_LAYERS  4
#define LAYER1      (size_t)  5, (size_t) 10, sigmoid
#define LAYER2      (size_t) 10, (size_t)  6, relu
#define LAYER3      (size_t)  6, (size_t)  4, softmax


int main(){
    std::normal_distribution<float> dist(-1, 1);
    NeuralNetwork nn (NUM_LAYERS, {new FCLayer(LAYER1), new FCLayer(LAYER2), new FCLayer(LAYER3)});
    NeuralNetwork nn2(NUM_LAYERS, {new FCLayer(LAYER1), new FCLayer(LAYER2), new FCLayer(LAYER3)});

    nn.printInfo();
    nn2.printInfo();

    Matrix<float> output = nn.feedforward(Matrix<float>(1, 5, dist));
    output.print();

    output = nn2.feedforward(Matrix<float>(1, 5, dist));
    output.print();

    return 0;
}
