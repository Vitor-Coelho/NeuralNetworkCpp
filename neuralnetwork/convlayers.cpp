#include "layers.hpp"
#include "activations.hpp"

std::normal_distribution<float> distConv(0,1);

ConvLayer::ConvLayer(size_t numFilters_, size_t filterRows, size_t filterCols, size_t stride_, bool padding_, std::string activation_){
    numFilters = numFilters_;
    filterRowSize = filterRows;
    filterColSize = filterCols;
    stride = stride_;
    padding = padding_;
    activation = strToFunc(activation_);
    actDerivative = strToDerivative(activation_);

    std::vector<Matrix<float>> filters;
    for(int i = 0; i < numFilters; i++){
        filters.push_back(Matrix<float>(filterRowSize, filterColSize, distConv));
    }
}

std::vector<Matrix<float>> ConvLayer::feedforward(std::vector<Matrix<float>> input){
    std::vector<Matrix<float>> result;
    for(auto in = input.begin(); in != input.end(); ++in){
        for(auto filter = filters.begin(); filter != filters.end(); ++filter){
            result.push_back(activation(in->filter(*filter, stride, padding)));
        }
    }
    return result;
}

// Helper function to allow feedforward with only one matrix
Matrix<float> ConvLayer::feedforward(Matrix<float> input){
    std::vector<Matrix<float>> input_;
    input_.push_back(input);
    return feedforward(input_).at(0);
}

std::vector<Matrix<float>> ConvLayer::feedWithMemory(std::vector<Matrix<float>> input){
    lastInput = input;

    lastPreAct.clear();
    for(auto in = input.begin(); in != input.end(); ++in){
        for(auto filter = filters.begin(); filter != filters.end(); ++filter){
            lastPreAct.push_back(in->filter(*filter, stride, padding));
        }
    }

    lastOutput.clear();
    for(auto pre = lastPreAct.begin(); pre != lastPreAct.end(); ++pre){
        lastOutput.push_back(activation(*pre));
    }

    return lastOutput;
}

// Matrix<float> ConvLayer::backpropagate(Matrix<float> error, float learningRate){
//     error = actDerivative(lastPreAct, error);
//     Matrix<float> gradient = lastInput.transpose() * error / lastInput.numRows();
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

std::vector<Matrix<float>> ConvLayer::getFilters(){return filters;}

std::string ConvLayer::getActivation(){return funcToStr(activation);}

void ConvLayer::setFilters(std::vector<Matrix<float>> newFilters){filters = newFilters;}

void ConvLayer::setActivation(std::string activation_){activation = strToFunc(activation_); actDerivative = strToDerivative(activation_);}


Matrix<float> ConvLayer::feedWithMemory(Matrix<float> input){throw std::invalid_argument("Convolutional layers can't train with matrices alone (must be vector of matrices)");}
Matrix<float> ConvLayer::backpropagate(Matrix<float> error, float learningRate){throw std::invalid_argument("Convolutional layers can't train with matrices alone (must be vector of matrices)");}
