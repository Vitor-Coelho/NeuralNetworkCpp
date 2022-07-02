#include <iostream>
#include "neuralnetwork/matrix.hpp"
#include "neuralnetwork/nn.hpp"
#include "neuralnetwork/layers.hpp"
#include "neuralnetwork/costs.hpp"
#include "neuralnetwork/dataset.hpp"

using namespace std;

#define NUM_LAYERS  4
#define LAYER1      (size_t)  5, (size_t) 10, sigmoid, sigmoidDerivative
#define LAYER2      (size_t) 10, (size_t)  6, relu   , reluDerivative
#define LAYER3      (size_t)  6, (size_t)  4, softmax, softmaxDerivative


int main(){
    std::normal_distribution<float> dist(-1, 1);
    NeuralNetwork nn (NUM_LAYERS, {new FCLayer(LAYER1), new FCLayer(LAYER2), new FCLayer(LAYER3)});

    Matrix<float> trainInput  = getMatrixFromCsv("../data/train_input.csv");
    Matrix<float> trainOutput = getMatrixFromCsv("../data/train_output.csv");

    Dataset dataset(1);
    dataset.setTrainInput(trainInput);
    dataset.setTrainOutput(trainOutput);
    dataset.print();

    nn.printInfo();

    Matrix<float> input(4, 5, dist);
    Matrix<float> target(4, 4);
    target.set(1, 0, 0);
    target.set(1, 1, 1);
    target.set(1, 2, 3);
    target.set(1, 3, 3);

    float error = nn.trainBatch(input, target, 0.05, crossEntropy, crossEntropyDerivative);
    cout << "Error: " << error << endl;

    error = nn.trainBatch(input, target, 0.05, crossEntropy, crossEntropyDerivative);
    cout << "Error: " << error << endl;

    error = nn.trainBatch(input, target, 0.05, crossEntropy, crossEntropyDerivative);
    cout << "Error: " << error << endl;

    return 0;
}
