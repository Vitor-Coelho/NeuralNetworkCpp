#include <iostream>
#include "neuralnetwork/matrix.hpp"
#include "neuralnetwork/nn.hpp"
#include "neuralnetwork/layers.hpp"
#include "neuralnetwork/costs.hpp"
#include "neuralnetwork/dataset.hpp"

using namespace std;

/* Deep network */
// #define NUM_LAYERS 3
// #define FCLAYER1   (size_t) 8, (size_t) 6, "relu"
// #define FCLAYER2   (size_t) 6, (size_t) 4, "softmax"

/* Conv net */
#define NUM_LAYERS 3
#define CONVLAYER1 32, 3, 3, 3, 1, true, "relu"
#define CONVLAYER2 16, 3, 3, 3, 1, true, "sigmoid"


int main(){
    std::normal_distribution<float> dist(-1, 1);
    // NeuralNetwork nn (NUM_LAYERS, {new FCLayer(FCLAYER1), new FCLayer(FCLAYER2)});
    NeuralNetwork nn (NUM_LAYERS, {new ConvLayer(CONVLAYER1), new ConvLayer(CONVLAYER2)});

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
