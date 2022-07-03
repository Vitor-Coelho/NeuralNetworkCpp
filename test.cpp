#include <iostream>
#include "neuralnetwork/matrix.hpp"
#include "neuralnetwork/nn.hpp"
#include "neuralnetwork/layers.hpp"
#include "neuralnetwork/costs.hpp"
#include "neuralnetwork/dataset.hpp"

using namespace std;

#define NUM_LAYERS  3
#define LAYER1      (size_t)  8, (size_t)  6, relu   , reluDerivative
#define LAYER2      (size_t)  6, (size_t)  4, softmax, softmaxDerivative


int main(){
    std::normal_distribution<float> dist(-1, 1);
    NeuralNetwork nn (NUM_LAYERS, {new FCLayer(LAYER1), new FCLayer(LAYER2)});

    Matrix<float> trainInput  = getMatrixFromCsv("../data/input.csv");
    Matrix<float> trainOutput = getMatrixFromCsv("../data/output.csv");

    Dataset dataset(50);
    dataset.setTrainInput(trainInput);
    dataset.setTrainOutput(trainOutput);
    dataset.print();

    nn.printInfo();

    nn.train(dataset, 500, 0.001, crossEntropy, crossEntropyDerivative);

    return 0;
}
