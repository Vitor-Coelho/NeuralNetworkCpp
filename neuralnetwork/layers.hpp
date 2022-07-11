#ifndef LAYERS_H
#define LAYERS_H

#include <random>
#include "matrix.hpp"
#include "activations.hpp"


/* Abstract layer class */
class Layer{    
    public:
        virtual Matrix<float> feedforward(Matrix<float> input) = 0;
        virtual Matrix<float> feedWithMemory(Matrix<float> input) = 0;
        virtual Matrix<float> backpropagate(Matrix<float> error, float learningRate) = 0;

        virtual std::vector<Matrix<float>> feedforward(std::vector<Matrix<float>> input) = 0;
        virtual std::vector<Matrix<float>> feedWithMemory(std::vector<Matrix<float>> input) = 0;
        virtual std::vector<Matrix<float>> backpropagate(std::vector<Matrix<float>> error, float learningRate) = 0;

        virtual void print() = 0;
        virtual void saveToFile() = 0;

        virtual size_t getInputSize() = 0;
        virtual size_t getOutputSize() = 0;
        virtual Matrix<float> getWeights() = 0;
        virtual std::string getActivation() = 0;

        virtual void setWeights(Matrix<float> newWeights) = 0;
        virtual void setActivation(std::string activation_) = 0;
};


/* Fully connected layer */
class FCLayer : public Layer{
    private:
        Matrix<float> weights;
        activation_t activation;
        act_deriv_t  actDerivative;
        Matrix<float> lastInput, lastPreAct, lastOutput;
        size_t inputSize, outputSize;
        bool bias = true;
    
    public:
        FCLayer(size_t inputLen, size_t outputLen, std::string activation_, bool addBias=true);
        Matrix<float> feedforward(Matrix<float> input);
        Matrix<float> feedWithMemory(Matrix<float> input);
        Matrix<float> backpropagate(Matrix<float> error, float learningRate);
        void print();
        void saveToFile();

        size_t getInputSize();
        size_t getOutputSize();
        Matrix<float> getWeights();
        std::string getActivation();

        void setWeights(Matrix<float> newWeights);
        void setActivation(std::string activation_);

        /* Unused functions for FCLayer (only throw exception) */
        std::vector<Matrix<float>> feedforward(std::vector<Matrix<float>> input);
        std::vector<Matrix<float>> feedWithMemory(std::vector<Matrix<float>> input);
        std::vector<Matrix<float>> backpropagate(std::vector<Matrix<float>> error, float learningRate);
};


/* Convolutional layer */
class ConvLayer : public Layer{
    private:
        Matrix<float> filter;
        activation_t activation;
        act_deriv_t  actDerivative;
        Matrix<float> lastInput, lastPreAct, lastOutput;
        size_t filterRowSize, filterColSize;
        bool bias = true;
    
    public:
        ConvLayer(size_t filterRows, size_t filterCols, std::string activation_, bool addBias=true);
        Matrix<float> feedforward(Matrix<float> input);
        std::vector<Matrix<float>> feedforward(std::vector<Matrix<float>> input);
        std::vector<Matrix<float>> feedWithMemory(std::vector<Matrix<float>> input);
        std::vector<Matrix<float>> backpropagate(std::vector<Matrix<float>> error, float learningRate); 

        void print();
        void saveToFile();

        size_t getfilterRowSize();
        size_t getfilterColSize();
        Matrix<float> getWeights();
        std::string getActivation();

        void setWeights(Matrix<float> newWeights);
        void setActivation(std::string activation_);

        /* Unused functions for ConvLayer (only throw exception) */
        Matrix<float> feedWithMemory(Matrix<float> input);
        Matrix<float> backpropagate(Matrix<float> error, float learningRate);  
};


class MaxPoolLayer : public Layer{
    // O erro só propaga aos neurônios que foram ativados (max)
};

#endif
