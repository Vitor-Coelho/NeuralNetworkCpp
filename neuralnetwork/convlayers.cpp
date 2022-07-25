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

// Helper function to allow feedforward with only one tensor
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

std::vector<Tensor3D<float>> ConvLayer::backpropagate(std::vector<Tensor3D<float>> error, float learningRate){
    /* Derivative of the activation function */
    for(size_t sample = 0; sample < error.size(); sample++){
        for(size_t ch = 0; ch < this->filterWidth; ch++){
            error.at(sample).set(actDerivative(lastPreAct.at(sample)(ch), error.at(sample)(ch)), ch);
        }
    }

    /* Update filters based on error */
    for(size_t filterIdx = 0; filterIdx < numFilters; filterIdx++){
        filters.at(filterIdx) = filters.at(filterIdx) - backpropKernel(error, filterIdx) * learningRate;
    }
    
    /* Update error for the next layer */
    // error = error * weights(1, weights.numRows()-1, 0, weights.numCols()-1).transpose();
    return error;
}

Tensor3D<float> ConvLayer::backpropKernel(std::vector<Tensor3D<float>> error, size_t numFilter){
    Tensor3D<float> kernelError;
    for(size_t sample = numFilter; sample < error.size(); sample += this->numFilters){
        for(size_t ch = 0; ch < this->filterWidth; ch++){
            kernelError(ch) += filterDeriv(error.at(sample)(ch), lastInput.at(sample)(ch), stride, padding);
        }
    }
    return kernelError;
}

Matrix<float> ConvLayer::filterDeriv(Matrix<float> error, Matrix<float> lastIn, size_t stride, bool padding){
    
    Matrix<float> input = lastIn;
    size_t rowSize, colSize;
    
    if(padding){
        int numPadRow = filterRowSize - 1;
        int numPadCol = filterColSize - 1;

        while(numPadRow){
            input = input.insert(Matrix<float>(1, input.numCols()), 0, ROW);
            input = input.append(Matrix<float>(1, input.numCols()), ROW);
            numPadRow -= 2;
        }

        while(numPadCol){
            input = input.insert(Matrix<float>(input.numRows(), 1), 0, COLUMN);
            input = input.append(Matrix<float>(input.numRows(), 1), COLUMN);
            numPadCol -= 2;
        }
    }

    Matrix<float> kernelDeriv(filterRowSize, filterColSize);

    for(size_t i = 0; i < error.numRows(); i++){
        for(size_t j = 0; j < error.numCols(); j++){
            kernelDeriv += lastIn(i*stride, i*stride + filterRowSize - 1, j*stride, j*stride + filterColSize - 1) * error(i, j);
        }
    }

    return kernelDeriv;
}

void ConvLayer::print(){
    std::cout << ">> Convolutional layer" << std::endl;
    std::cout << "Stride: " << stride << " | Padding: " << padding << std::endl;
    std::cout << "Filters:";
    for(auto filter = filters.begin(); filter != filters.end(); ++filter){
        filter->print();
    }
}

void ConvLayer::saveToFile(std::string path, int idx){}

size_t ConvLayer::getfilterRowSize(){return filterRowSize;}

size_t ConvLayer::getfilterColSize(){return filterColSize;}

std::vector<Tensor3D<float>> ConvLayer::getFilters(){return filters;}

std::string ConvLayer::getActivation(){return funcToStr(activation);}

void ConvLayer::setFilters(std::vector<Tensor3D<float>> newFilters){filters = newFilters;}

void ConvLayer::setActivation(std::string activation_){activation = strToFunc(activation_); actDerivative = strToDerivative(activation_);}

Matrix<float> ConvLayer::feedforward(Matrix<float> input){throw std::invalid_argument("Convolutional layers can't train with matrices alone (must be vector of 3D tensors)");}
Matrix<float> ConvLayer::feedWithMemory(Matrix<float> input){throw std::invalid_argument("Convolutional layers can't train with matrices alone (must be vector of 3D tensors)");}
Matrix<float> ConvLayer::backpropagate(Matrix<float> error, float learningRate){throw std::invalid_argument("Convolutional layers can't train with matrices alone (must be vector of 3D tensors)");}
