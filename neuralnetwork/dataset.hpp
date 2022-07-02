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
        void writeDatasetToCsv(std::string path);
        static Dataset getMatrixFromCsv(std::string path);
        void printInfo();       // Print dataset information (number of samples; pct of train, test, val; first five samples of train)
        size_t getNumSamples();

        Matrix<float> getBatchInput();
        Matrix<float> getBatchOutput();

        Matrix<float> getTrainInput(){return trainInput;};
        Matrix<float> getTrainOutput(){return trainOutput;};
        Matrix<float> getTestInput(){return testInput;};
        Matrix<float> getTestOutput(){return testOutput;};
        Matrix<float> getValidationInput(){return validationInput;};
        Matrix<float> getValidationOutput(){return validationOutput;};

        void setBatchSize();
        Matrix<float> setTrainInput();
        Matrix<float> setTrainOutput();
        Matrix<float> setTestInput();
        Matrix<float> setTestOutput();
        Matrix<float> setValidationInput();
        Matrix<float> setValidationOutput();
};

inline Dataset::Dataset(size_t batchSize_){
    batchSize = batchSize_;
    dataIdx = 0;
}

inline Dataset::Dataset(Matrix<float> input, Matrix<float> output, float trainPct, float testPct, float validationPct){

}

inline bool Dataset::nextBatch(){
    dataIdx += batchSize;
    if(dataIdx >= trainInput.numRows()){
        dataIdx = 0;
        return false;   // Data is over
    }
    return true;    // There is more data
}

inline Matrix<float> Dataset::getBatchInput(){
    size_t dataEnd = dataIdx + batchSize < trainInput.numRows() ? dataIdx + batchSize : trainInput.numRows() - 1;
    return trainInput(dataIdx, dataEnd, 0, trainInput.numCols()-1);
}

inline Matrix<float> Dataset::getBatchOutput(){
    size_t dataEnd = dataIdx + batchSize < trainOutput.numRows() ? dataIdx + batchSize : trainOutput.numRows() - 1;
    return trainOutput(dataIdx, dataEnd, 0, trainOutput.numCols()-1);
}

#endif
