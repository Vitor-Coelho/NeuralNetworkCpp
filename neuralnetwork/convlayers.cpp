#include "layers.hpp"
#include "activations.hpp"
#include "tensor3d.hpp"

std::uniform_real_distribution<float> distConv(-1,1);

ConvLayer::ConvLayer(size_t numFilters_, size_t filterRows, size_t filterCols, size_t width, size_t stride_, bool padding_, std::string activation_){
    numFilters = numFilters_;
    filterRowSize = filterRows;
    filterColSize = filterCols;
    filterWidth = width;
    stride = stride_;
    padding = padding_;
    activation = strToFunc(activation_);
    actDerivative = strToDerivative(activation_);

    std::vector<Tensor3D<float>> filters;
    for(int i = 0; i < numFilters; i++){
        filters.push_back(Tensor3D<float>(filterRowSize, filterColSize, filterWidth, distConv));
    }
}

std::vector<Tensor3D<float>> ConvLayer::feedforward(std::vector<Tensor3D<float>> input){
    std::vector<Tensor3D<float>> result;
    for(auto in = input.begin(); in != input.end(); ++in){
        for(auto filter = filters.begin(); filter != filters.end(); ++filter){
            result.push_back(in->filter(*filter, stride, padding).applyFunction(activation));
        }
    }
    return result;
}

// Helper function to allow feedforward with only one matrix
Tensor3D<float> ConvLayer::feedforward(Tensor3D<float> input){
    std::vector<Tensor3D<float>> input_;
    input_.push_back(input);
    return feedforward(input_).at(0);
}

std::vector<Tensor3D<float>> ConvLayer::feedWithMemory(std::vector<Tensor3D<float>> input){
    lastInput = input;

    lastPreAct.clear();
    for(auto in = input.begin(); in != input.end(); ++in){
        for(auto filter = filters.begin(); filter != filters.end(); ++filter){
            lastPreAct.push_back(in->filter(*filter, stride, padding));
        }
    }

    lastOutput.clear();
    for(auto pre = lastPreAct.begin(); pre != lastPreAct.end(); ++pre){
        lastOutput.push_back(pre->applyFunction(activation));
    }

    return lastOutput;
}

// Tensor3D<float> ConvLayer::backpropagate(Tensor3D<float> error, float learningRate){
//     error = actDerivative(lastPreAct, error);
//     Tensor3D<float> gradient = lastInput.transpose() * error / lastInput.numRows();
//     weights -= gradient*learningRate;
//     error = error * weights(1, weights.numRows()-1, 0, weights.numCols()-1).transpose();
//     return error;
// }

void ConvLayer::print(){
    std::cout << ">> Convolutional layer" << std::endl;
    std::cout << "Stride: " << stride << " | Padding: " << padding << std::endl;
    std::cout << "Filters:";
    for(auto filter = filters.begin(); filter != filters.end(); ++filter){
        filter->print();
    }
}

void ConvLayer::saveToFile(){}

size_t ConvLayer::getfilterRowSize(){return filterRowSize;}

size_t ConvLayer::getfilterColSize(){return filterColSize;}

std::vector<Tensor3D<float>> ConvLayer::getFilters(){return filters;}

std::string ConvLayer::getActivation(){return funcToStr(activation);}

void ConvLayer::setFilters(std::vector<Tensor3D<float>> newFilters){filters = newFilters;}

void ConvLayer::setActivation(std::string activation_){activation = strToFunc(activation_); actDerivative = strToDerivative(activation_);}

Matrix<float> ConvLayer::feedforward(Matrix<float> input){throw std::invalid_argument("Convolutional layers can't train with matrices alone (must be vector of 3D tensors)");}
Matrix<float> ConvLayer::feedWithMemory(Matrix<float> input){throw std::invalid_argument("Convolutional layers can't train with matrices alone (must be vector of 3D tensors)");}
Matrix<float> ConvLayer::backpropagate(Matrix<float> error, float learningRate){throw std::invalid_argument("Convolutional layers can't train with matrices alone (must be vector of 3D tensors)");}
