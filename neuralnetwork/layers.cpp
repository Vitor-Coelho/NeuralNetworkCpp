#include "layers.hpp"

std::normal_distribution<float> dist(0,1);

FCLayer::FCLayer(size_t inputLen, size_t outputLen, Matrix<float> (*act) (Matrix<float>), bool addBias){
    inputSize = inputLen;
    outputSize = outputLen;
    activation = act;
    bias = addBias;

    size_t rows = 0;
    bias ? rows = inputSize + 1 : rows = inputSize;

    weights = Matrix<float>(rows, outputSize, dist);
}

Matrix<float> FCLayer::feedforward(Matrix<float> input){
    if(bias)
        input = input.insert(Matrix<float>(1, 1, (std::string) "ones"), 0, COLUMN);  // Add bias
    return activation(input * weights);
}

void FCLayer::print(){std::cout << ">> Fully connected layer"; weights.print();}

void FCLayer::saveToFile(){}
