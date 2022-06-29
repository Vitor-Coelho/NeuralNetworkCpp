#ifndef DATASET_H
#define DATASET_H

#include <string>
#include "matrix.hpp"

// TODO: definir os métodos da classe Dataset
// • nos métodos que envolvem CSV, colocar o path como o caminho + nome de arquivo, 
//   mas os arquivos em si terão um sufixo de _trainIn, _trainOut... a depender de 
//   quais destas separações o dataset possui

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
        Dataset();
        void nextBatch();       // Update data index to the next batch
        void setBatchSize();
        void shuffle();         // Shuffle train set
        Matrix<float> getTrainInput();
        Matrix<float> getTrainOutput();

        void writeDatasetToCsv(std::string path);
        static Dataset getDatasetFromCsv(std::string path);
};

void Dataset::nextBatch(){

}

#endif