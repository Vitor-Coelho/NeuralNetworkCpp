#ifndef DATASET_H
#define DATASET_H

#include <string>
#include "matrix.hpp"

// TODO: definir os métodos da classe Dataset
// • nos métodos que envolvem CSV, colocar o path como o caminho + nome de arquivo, 
//   mas os arquivos em si terão um sufixo de _trainIn, _trainOut... a depender de 
//   quais destas separações o dataset possui
// • nos métodos de se definir as matrizes, deve-se sempre iniciar colocando os inputs
//   e depois os outputs, pois a qtd de linhas será testada nas definições de output

class Dataset{
    private:
        Matrix<float> trainInput;
        Matrix<float> trainOutput;
        Matrix<float> testInput;
        Matrix<float> testOutput;
        Matrix<float> validationInput;
        Matrix<float> validationOutput;
        size_t batchSize;
        size_t dataIdx;

    public:
        Dataset(size_t batchSize_);
        Dataset(Matrix<float> input, Matrix<float> output, float trainPct, float testPct, float validationPct);

        bool nextBatch();       // Update data index to the next batch - returns true if all data has been done
        void shuffle();         // Shuffle train set
        void writeToCsv(std::string path);
        static Matrix<float> getMatrixFromCsv(std::string path);
        static Dataset getDatasetFromCsv(std::string path);
        void printInfo();       // Print dataset information (number of samples; pct of train, test, val; first five samples of train)
        size_t getNumSamples(){return trainInput.numRows();};

        Matrix<float> getBatchInput();
        Matrix<float> getBatchOutput();

        Matrix<float> getTrainInput(){return trainInput;};
        Matrix<float> getTrainOutput(){return trainOutput;};
        Matrix<float> getTestInput(){return testInput;};
        Matrix<float> getTestOutput(){return testOutput;};
        Matrix<float> getValidationInput(){return validationInput;};
        Matrix<float> getValidationOutput(){return validationOutput;};

        void setBatchSize(size_t batchSize_){batchSize = batchSize_;};
        Matrix<float> setTrainInput(Matrix<float> data);
        Matrix<float> setTrainOutput(Matrix<float> data);
        Matrix<float> setTestInput(Matrix<float> data);
        Matrix<float> setTestOutput(Matrix<float> data);
        Matrix<float> setValidationInput(Matrix<float> data);
        Matrix<float> setValidationOutput(Matrix<float> data);
};

inline Dataset::Dataset(size_t batchSize_){
    batchSize = batchSize_;
    dataIdx = 0;
}

inline Dataset::Dataset(Matrix<float> input, Matrix<float> output, float trainPct, float testPct, float validationPct){
    size_t trainIdx      = (size_t) trainPct*input.numRows();
    size_t testIdx       = (size_t) testPct*input.numRows() + trainIdx;
    size_t validationIdx = (size_t) validationPct*input.numRows() + testIdx;

    trainInput  =  input(0, trainIdx, 0, input.numCols()-1);
    trainOutput = output(0, trainIdx, 0, input.numCols()-1);
    testInput   =  input(trainIdx, testIdx, 0, input.numCols()-1);
    testOutput  = output(trainIdx, testIdx, 0, input.numCols()-1);
    validationInput  =  input(testIdx, validationIdx, 0, input.numCols()-1);
    validationOutput = output(testIdx, validationIdx, 0, input.numCols()-1);
}

inline bool Dataset::nextBatch(){
    dataIdx += batchSize;
    if(dataIdx >= trainInput.numRows()){
        dataIdx = 0;
        return false;   // Data is over
    }
    return true;    // There is more data
}

inline void Dataset::shuffle(){
    size_t size = trainInput.numRows();
    std::vector<size_t> idxList(size);
    Matrix<float> oldInput = trainInput, oldOutput = trainOutput;

    std::generate(idxList.begin(), idxList.end(), [n=0]()mutable{return n++;});
    std::shuffle(idxList.begin(), idxList.end(), mt);

    for(size_t i = 0; i < size; i++){
        trainInput.set(oldInput(idxList.at(i), ROW), i, ROW);
        trainOutput.set(oldOutput(idxList.at(i), ROW), i, ROW);
    }
}

void writeDatasetToCsv(std::string path);
static Dataset getMatrixFromCsv(std::string path);
static Dataset getDatasetFromCsv(std::string path);

inline Matrix<float> Dataset::getBatchInput(){
    size_t dataEnd = dataIdx + batchSize < trainInput.numRows() ? dataIdx + batchSize : trainInput.numRows() - 1;
    return trainInput(dataIdx, dataEnd, 0, trainInput.numCols()-1);
}

inline Matrix<float> Dataset::getBatchOutput(){
    size_t dataEnd = dataIdx + batchSize < trainOutput.numRows() ? dataIdx + batchSize : trainOutput.numRows() - 1;
    return trainOutput(dataIdx, dataEnd, 0, trainOutput.numCols()-1);
}

inline Matrix<float> Dataset::setTrainInput(Matrix<float> data){
    trainInput = data;
}

inline Matrix<float> Dataset::setTrainOutput(Matrix<float> data){
    if(data.numRows() != trainInput.numRows())
        throw std::invalid_argument("Number of samples different from input data");
    trainOutput = data;
}

inline Matrix<float> Dataset::setTestInput(Matrix<float> data){
    testInput = data;
}

inline Matrix<float> Dataset::setTestOutput(Matrix<float> data){
    if(data.numRows() != testInput.numRows())
        throw std::invalid_argument("Number of samples different from input data");
    testOutput = data;
}

inline Matrix<float> Dataset::setValidationInput(Matrix<float> data){
    validationInput = data;
}

inline Matrix<float> Dataset::setValidationOutput(Matrix<float> data){
    if(data.numRows() != validationInput.numRows())
        throw std::invalid_argument("Number of samples different from input data");
    validationOutput = data;
}

#endif
