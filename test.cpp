#include <iostream>
#include "neuralnetwork/matrix.hpp"
#include "neuralnetwork/nn.hpp"
#include "neuralnetwork/layers.hpp"
#include "neuralnetwork/costs.hpp"
#include "neuralnetwork/dataset.hpp"

using namespace std;

#define NUM_LAYERS  3
#define LAYER1      (size_t)  8, (size_t)  6, "relu"
#define LAYER2      (size_t)  6, (size_t)  4, "softmax"


int main(){
    std::normal_distribution<float> dist(-1, 1);
    NeuralNetwork nn (NUM_LAYERS, {new FCLayer(LAYER1), new FCLayer(LAYER2)});

    Matrix<float> input  = getMatrixFromCsv("../data/input.csv");
    Matrix<float> output = getMatrixFromCsv("../data/output.csv");

    Dataset dataset(input, output, 300, 0.85, 0.15, 0);
    dataset.shuffle();
    dataset.print();

    nn.printInfo();

    nn.train(dataset, 750, 0.1, crossEntropy, crossEntropyDerivative);

    nn.printInfo();

    nn.saveToFile("../data/neural/");

    nn = NeuralNetwork::fromFile("../data/neural/");

    nn.printInfo();

    return 0;
}
