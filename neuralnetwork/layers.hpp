#ifndef LAYERS_H
#define LAYERS_H

#include <random>
#include "matrix.hpp"


/* Abstract layer class */
class Layer{    
    public:
        virtual Matrix<float> feedforward(Matrix<float> input) = 0;
        virtual void print() = 0;
        virtual void saveToFile() = 0;
};


/* Fully connected layer */
class FCLayer : public Layer{
    private:
        Matrix<float> weights;
        Matrix<float> (*activation) (Matrix<float>);
        size_t inputSize, outputSize;
        bool bias = true;
    
    public:
        FCLayer(size_t inputLen, size_t outputLen, Matrix<float> (*act) (Matrix<float>), bool addBias=true);
        Matrix<float> feedforward(Matrix<float> input);
        void print();
        void saveToFile();
};


class ConvLayer : public Layer{
    
};


class MaxPoolLayer : public Layer{
    
};

#endif
