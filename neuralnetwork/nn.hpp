#ifndef NN_H
#define NN_H

#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include "matrix.hpp"
#include "activations.hpp"
#include "layers.hpp"

// Entrada: vetor linha (Matrix(1,n))


class NeuralNetwork{

    private:
        size_t numLayers, inputSize, outputSize;
        std::vector<Layer*> layers;

    public:
        NeuralNetwork(size_t numberOfLayers, std::vector<Layer*> NNLayers);
        ~NeuralNetwork();

        void printInfo();

        int getNumLayers();
        std::vector<size_t> getLayerSizes();
        std::vector<Layer*> getLayers();

        void assignNN(NeuralNetwork nn);
        void operator=(NeuralNetwork nn);

        Matrix<float> feedforward(Matrix<float> input);
        void saveToFile(std::string path);
};


NeuralNetwork::NeuralNetwork(size_t numberOfLayers, std::vector<Layer*> NNLayers){
    if(numberOfLayers != NNLayers.size() + 1)
        throw std::invalid_argument("Number of layers do not match");

    numLayers = numberOfLayers;
    layers = NNLayers;
    std::copy(NNLayers.begin(), NNLayers.end(), layers.begin());
}

NeuralNetwork::~NeuralNetwork(){}


void NeuralNetwork::printInfo(){
    std::cout << std::endl << std::endl << "Number of layers: " << numLayers << "\n\n*** Layers ***\n\n";
    for(auto layer = layers.begin(); layer != layers.end(); ++layer)
        (*layer)->print();
}

int NeuralNetwork::getNumLayers(){return numLayers;}
std::vector<Layer*> NeuralNetwork::getLayers(){return layers;}

void NeuralNetwork::assignNN(NeuralNetwork nn){
    numLayers = nn.getNumLayers();
    layers = nn.getLayers();
}

void NeuralNetwork::operator=(NeuralNetwork nn){
    this->assignNN(nn);
}

Matrix<float> NeuralNetwork::feedforward(Matrix<float> input){
    for(auto layer = layers.begin(); layer != layers.end(); ++layer){
        input = (*layer)->feedforward(input);
    }
    return input;
}

void NeuralNetwork::saveToFile(std::string path){} // TODO

#endif
