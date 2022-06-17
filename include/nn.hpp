#ifndef NN_H
#define NN_H

#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include "matrix.hpp"

// Entrada: vetor linha (Matrix(1,n))

Matrix sigmoid(Matrix x);
Matrix relu(Matrix x);
Matrix softmax(Matrix x);

class NeuralNetwork{

    private:
        int numLayers;
        std::vector<size_t> layerSizes;
        std::vector<Matrix*> weights;
        std::vector<Matrix(*)(Matrix)> activations;

    public:
        NeuralNetwork(int layers, std::vector<size_t> sizes, bool addBias=true);
        NeuralNetwork(int layers, std::vector<size_t> sizes, std::vector<Matrix(*)(Matrix)> act, bool addBias=true);
        NeuralNetwork(int layers, std::vector<size_t> sizes, std::vector<Matrix> w,
                      std::vector<Matrix(*)(Matrix)> act);
        ~NeuralNetwork();

        void printWeights();

        std::vector<size_t> getLayers();
        std::vector<Matrix*> getWeights();
        std::vector<Matrix(*)(Matrix)> getActivations();

        Matrix feedforward(Matrix input, bool addBias=true);
        void saveToFile(std::string path);

};

#endif
