#ifndef LAYERS_H
#define LAYERS_H

#include <random>
#include "matrix.hpp"
#include "tensor3d.hpp"
#include "activations.hpp"


/* Abstract layer class */
class Layer{    
    public:
        virtual Matrix<float> feedforward(Matrix<float> input) = 0;
        virtual Matrix<float> feedWithMemory(Matrix<float> input) = 0;
        virtual Matrix<float> backpropagate(Matrix<float> error, float learningRate) = 0;

        virtual Tensor3D<float> feedforward(Tensor3D<float> input) = 0;
        virtual std::vector<Tensor3D<float>> feedforward(std::vector<Tensor3D<float>> input) = 0;
        virtual std::vector<Tensor3D<float>> feedWithMemory(std::vector<Tensor3D<float>> input) = 0;
        virtual std::vector<Tensor3D<float>> backpropagate(std::vector<Tensor3D<float>> error, float learningRate) = 0;

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
        size_t inputSize, outputSize;
        bool bias = true;

        activation_t activation;
        act_deriv_t  actDerivative;

        Matrix<float> lastInput, lastPreAct, lastOutput;
    
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
        Tensor3D<float> feedforward(Tensor3D<float> input);
        std::vector<Tensor3D<float>> feedforward(std::vector<Tensor3D<float>> input);
        std::vector<Tensor3D<float>> feedWithMemory(std::vector<Tensor3D<float>> input);
        std::vector<Tensor3D<float>> backpropagate(std::vector<Tensor3D<float>> error, float learningRate);
};


/* Convolutional layer */
class ConvLayer : public Layer{
    private:
        std::vector<Tensor3D<float>> filters;
        size_t numFilters, filterRowSize, filterColSize, filterWidth, stride;
        bool padding;

        activation_t activation;
        act_deriv_t  actDerivative;

        std::vector<Tensor3D<float>> lastInput, lastPreAct, lastOutput;
        
    public:
        ConvLayer(size_t numFilters_, size_t filterRows, size_t filterCols, size_t width, size_t stride_, bool padding_, std::string activation_);
        Tensor3D<float> feedforward(Tensor3D<float> input);
        std::vector<Tensor3D<float>> feedforward(std::vector<Tensor3D<float>> input);
        std::vector<Tensor3D<float>> feedWithMemory(std::vector<Tensor3D<float>> input);
        std::vector<Tensor3D<float>> backpropagate(std::vector<Tensor3D<float>> error, float learningRate); 

        void print();
        void saveToFile();

        size_t getfilterRowSize();
        size_t getfilterColSize();
        std::vector<Tensor3D<float>> getFilters();
        std::string getActivation();

        void setFilters(std::vector<Tensor3D<float>> newFilters);
        void setActivation(std::string activation_);

        /* Unused functions for ConvLayer (only throw exception) */
        Matrix<float> feedforward(Matrix<float> input);
        Matrix<float> feedWithMemory(Matrix<float> input);
        Matrix<float> backpropagate(Matrix<float> error, float learningRate);  
};


class MaxPoolLayer : public Layer{
    // O erro só propaga aos neurônios que foram ativados (max)
};

#endif
