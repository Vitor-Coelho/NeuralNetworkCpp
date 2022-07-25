#include "layers.hpp"
#include "activations.hpp"

std::normal_distribution<float> dist(0,1);

FCLayer::FCLayer(size_t inputLen, size_t outputLen, std::string activation_, bool addBias){
    inputSize = inputLen;
    outputSize = outputLen;
    activation = strToFunc(activation_);
    actDerivative = strToDerivative(activation_);
    bias = addBias;

    size_t rows = 0;
    bias ? rows = inputSize + 1 : rows = inputSize;

    weights = Matrix<float>(rows, outputSize, dist);
}

Matrix<float> FCLayer::feedforward(Matrix<float> input){
    if(bias)
        input = input.insert(Matrix<float>(input.numRows(), 1, (std::string) "ones"), 0, COLUMN);  // Add bias
    return activation(input * weights);
}

Matrix<float> FCLayer::feedWithMemory(Matrix<float> input){
    if(bias)
        input = input.insert(Matrix<float>(input.numRows(), 1, (std::string) "ones"), 0, COLUMN);  // Add bias
    lastInput = input;
    lastPreAct = input * weights;
    lastOutput = activation(lastPreAct);
    return lastOutput;
}

Matrix<float> FCLayer::backpropagate(Matrix<float> error, float learningRate){
    error = actDerivative(lastPreAct, error);
    Matrix<float> gradient = lastInput.transpose() * error / lastInput.numRows();
    weights -= gradient*learningRate;
    error = error * weights(1, weights.numRows()-1, 0, weights.numCols()-1).transpose();
    return error;
}

void FCLayer::print(){std::cout << ">> Fully connected layer"; weights.print();}

void FCLayer::saveToFile(){}

size_t FCLayer::getInputSize(){return inputSize;}

size_t FCLayer::getOutputSize(){return outputSize;}

Matrix<float> FCLayer::getWeights(){return weights;}

std::string FCLayer::getActivation(){return funcToStr(activation);}

void FCLayer::setWeights(Matrix<float> newWeights){weights = newWeights;}

void FCLayer::setActivation(std::string activation_){activation = strToFunc(activation_); actDerivative = strToDerivative(activation_);}

Tensor3D<float> FCLayer::feedforward(Tensor3D<float> input){throw std::invalid_argument("Fully connected layers can't get vector of matrices as inputs");}
std::vector<Tensor3D<float>> FCLayer::feedforward(std::vector<Tensor3D<float>> input){throw std::invalid_argument("Fully connected layers can't get vector of matrices as inputs");}
std::vector<Tensor3D<float>> FCLayer::feedWithMemory(std::vector<Tensor3D<float>> input){throw std::invalid_argument("Fully connected layers can't get vector of matrices as inputs");}
std::vector<Tensor3D<float>> FCLayer::backpropagate(std::vector<Tensor3D<float>> error, float learningRate){throw std::invalid_argument("Fully connected layers can't get vector of matrices as inputs");}
