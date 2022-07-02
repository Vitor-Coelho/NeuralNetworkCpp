#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <fstream>
#include <sstream>
#include "matrix.hpp"

#define DELIMITER ","


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
        void print();   // Print dataset information (num samples; pct of train, test, val; first five samples of train)
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
        void setTrainInput(Matrix<float> data);
        void setTrainOutput(Matrix<float> data);
        void setTestInput(Matrix<float> data);
        void setTestOutput(Matrix<float> data);
        void setValidationInput(Matrix<float> data);
        void setValidationOutput(Matrix<float> data);

        void operator=(Dataset toCopy);
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
    std::shuffle(idxList.begin(), idxList.end(), mt());

    for(size_t i = 0; i < size; i++){
        trainInput.set(oldInput(idxList.at(i), ROW), i, ROW);
        trainOutput.set(oldOutput(idxList.at(i), ROW), i, ROW);
    }
}

inline void Dataset::writeToCsv(std::string path){
    std::fstream fout;

    fout.open(path + "_intrain", std::ios::out | std::ios::trunc);
    for(size_t i = 0; i < trainInput.numRows(); i++){
        for(size_t j = 0; j < trainInput.numCols(); j++){
            fout << trainInput(i,j);
            if(j != trainInput.numCols()-1)
                fout << DELIMITER;
        }
        fout << "\n";
    }

    fout.open(path + "_outtrain", std::ios::out | std::ios::trunc);
    for(size_t i = 0; i < trainOutput.numRows(); i++){
        for(size_t j = 0; j < trainOutput.numCols(); j++){
            fout << trainOutput(i,j);
            if(j != trainOutput.numCols()-1)
                fout << DELIMITER;
        }
        fout << "\n";
    }

    fout.open(path + "_intest", std::ios::out | std::ios::trunc);
    for(size_t i = 0; i < testInput.numRows(); i++){
        for(size_t j = 0; j < testInput.numCols(); j++){
            fout << testInput(i,j);
            if(j != testInput.numCols()-1)
                fout << DELIMITER;
        }
        fout << "\n";
    }

    fout.open(path + "_outtest", std::ios::out | std::ios::trunc);
    for(size_t i = 0; i < testOutput.numRows(); i++){
        for(size_t j = 0; j < testOutput.numCols(); j++){
            fout << testOutput(i,j);
            if(j != testOutput.numCols()-1)
                fout << DELIMITER;
        }
        fout << "\n";
    }

    fout.open(path + "_inval", std::ios::out | std::ios::trunc);
    for(size_t i = 0; i < validationInput.numRows(); i++){
        for(size_t j = 0; j < validationInput.numCols(); j++){
            fout << validationInput(i,j);
            if(j != validationInput.numCols()-1)
                fout << DELIMITER;
        }
        fout << "\n";
    }

    fout.open(path + "_outval", std::ios::out | std::ios::trunc);
    for(size_t i = 0; i < validationOutput.numRows(); i++){
        for(size_t j = 0; j < validationOutput.numCols(); j++){
            fout << validationOutput(i,j);
            if(j != validationOutput.numCols()-1)
                fout << DELIMITER;
        }
        fout << "\n";
    }
}

inline static Matrix<float> getMatrixFromCsv(std::string path){
    std::ifstream fin;

    Matrix<float> data;
    std::vector<float> row;
  
    fin.open(path, std::ios::in);

    std::string line, word, temp;
  
    while(std::getline(fin, line)){
        row.clear();
        std::stringstream s(line);

        while(std::getline(s, word, ',')){
            row.push_back(std::stof(word));
        }

        data = data.append(Matrix<float>(row, (size_t) 1, row.size()), ROW);
    }

    return data;
}

inline static Dataset getDatasetFromCsv(std::string path){
    Dataset dataset(1);
    
    dataset.setTrainInput(getMatrixFromCsv(path + "_intrain"));
    dataset.setTrainOutput(getMatrixFromCsv(path + "_outtrain"));
    dataset.setTestInput(getMatrixFromCsv(path + "_intest"));
    dataset.setTestOutput(getMatrixFromCsv(path + "_outtest"));
    dataset.setValidationInput(getMatrixFromCsv(path + "_inval"));
    dataset.setValidationOutput(getMatrixFromCsv(path + "_outval"));

    return dataset;
}

inline void Dataset::print(){
    size_t trainSize, testSize, valSize, totalSize;
    trainSize = trainInput.numRows();
    testSize = testInput.numRows();
    valSize = validationInput.numRows();
    totalSize = trainSize + testSize + valSize;

    std::cout << std::endl << "Number of samples: " << totalSize << std::endl;
    std::cout << "Train samples: " << trainSize << " (" << trainSize/totalSize*100 << "%)" << std::endl;
    std::cout << "Test samples: " << testSize << " (" << testSize/totalSize*100 << "%)" << std::endl;
    std::cout << "Validation samples: " << valSize << " (" << valSize/totalSize*100 << "%)" << std::endl;

    size_t printIdx = trainSize > 5 ? 5 : trainSize-1;
    trainInput(0, printIdx, 0, trainInput.numCols()-1).print();
    trainOutput(0, printIdx, 0, trainOutput.numCols()-1).print();
}

inline Matrix<float> Dataset::getBatchInput(){
    size_t dataEnd = dataIdx + batchSize < trainInput.numRows() ? dataIdx + batchSize : trainInput.numRows() - 1;
    return trainInput(dataIdx, dataEnd, 0, trainInput.numCols()-1);
}

inline Matrix<float> Dataset::getBatchOutput(){
    size_t dataEnd = dataIdx + batchSize < trainOutput.numRows() ? dataIdx + batchSize : trainOutput.numRows() - 1;
    return trainOutput(dataIdx, dataEnd, 0, trainOutput.numCols()-1);
}

inline void Dataset::setTrainInput(Matrix<float> data){
    trainInput = data;
}

inline void Dataset::setTrainOutput(Matrix<float> data){
    if(data.numRows() != trainInput.numRows())
        throw std::invalid_argument("Number of samples different from input data");
    trainOutput = data;
}

inline void Dataset::setTestInput(Matrix<float> data){
    testInput = data;
}

inline void Dataset::setTestOutput(Matrix<float> data){
    if(data.numRows() != testInput.numRows())
        throw std::invalid_argument("Number of samples different from input data");
    testOutput = data;
}

inline void Dataset::setValidationInput(Matrix<float> data){
    validationInput = data;
}

inline void Dataset::setValidationOutput(Matrix<float> data){
    if(data.numRows() != validationInput.numRows())
        throw std::invalid_argument("Number of samples different from input data");
    validationOutput = data;
}

void Dataset::operator=(Dataset toCopy){
    this->trainInput = toCopy.trainInput;
    this->trainOutput = toCopy.trainOutput;
    this->testInput = toCopy.testInput;
    this->testOutput = toCopy.testOutput;
    this->validationInput = toCopy.validationInput;
    this->validationOutput = toCopy.validationOutput;
}

#endif
