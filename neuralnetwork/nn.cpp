#include "nn.hpp"


NeuralNetwork::NeuralNetwork(size_t numberOfLayers){
    numLayers = numberOfLayers;
    for(size_t i = 0; i < numberOfLayers - 1; i++)
        layers.push_back(new FCLayer(0,0,"sigmoid"));
}

NeuralNetwork::NeuralNetwork(size_t numberOfLayers, std::vector<Layer*> NNLayers){
    if(numberOfLayers != NNLayers.size() + 1)
        throw std::invalid_argument("Number of layers do not match");

    numLayers = numberOfLayers;
    layers.assign(NNLayers.begin(), NNLayers.end());
}

NeuralNetwork::~NeuralNetwork(){}


void NeuralNetwork::printInfo(){
    std::cout << std::endl << std::endl << "Number of layers: " << numLayers << "\n\n*** Layers ***\n\n";
    for(auto layer = layers.begin(); layer != layers.end(); ++layer)
        (*layer)->print();
}

int NeuralNetwork::getNumLayers(){return numLayers;}
std::vector<Layer*> NeuralNetwork::getLayers(){return layers;}

void NeuralNetwork::assignNN(NeuralNetwork nn){ // Test references
    numLayers = nn.getNumLayers();

    layers.resize(nn.getLayers().size());
    for(size_t idx = 0; idx < layers.size(); idx++){
        layers.at(idx)->setLayer(nn.getLayers().at(idx));
    }
}

void NeuralNetwork::operator=(NeuralNetwork nn){
    this->assignNN(nn);
}

Matrix<float> NeuralNetwork::feedforward(Matrix<float> input){
    for(auto layer = layers.begin(); layer != layers.end(); ++layer){
        input = (*layer)->feedforward(input);
    }
    return input;
}

void NeuralNetwork::train(Dataset dataset, int epochs, float learningRate, cost_t costFunc, cost_deriv_t costDer){
    while(epochs--){
        float trainError = 0, testError;

        while(true){
            trainError += trainBatch(dataset.getBatchInput(), dataset.getBatchOutput(), learningRate, costFunc, costDer);
            if(!dataset.nextBatch())
                break;
        }
        
        testError = costFunc(feedforward(dataset.getTestInput()), dataset.getTestOutput());
        dataset.shuffle();
        
        std::cout << "Epoch " << epochs << " | Train error: " << trainError << " | Test error: " << testError << std::endl;
    }
}

float NeuralNetwork::trainBatch(Matrix<float> input, Matrix<float> target, float learningRate, cost_t costFunc, cost_deriv_t costDer){
    if(input.numRows() != target.numRows())
        throw std::invalid_argument("Input and targets have different number of rows");

    for(auto layer = layers.begin(); layer != layers.end(); ++layer){
        input = (*layer)->feedWithMemory(input);
    }

    float error = costFunc(input, target);
    Matrix<float> errorMatrix = costDer(input, target);

    for(auto layer = layers.rbegin(); layer != layers.rend(); ++layer){
        errorMatrix = (*layer)->backpropagate(errorMatrix, learningRate);
    }

    return error;
}

void NeuralNetwork::saveToFile(std::string path){
    std::fstream fout;
    fout << std::fixed << std::setprecision(10);
    fout.open(path + "info.txt", std::ios::out | std::ios::trunc);
    fout << this->getNumLayers() << "\n";
    fout.close();
    fout.clear();

    int idx = 0;
    for(auto layer = layers.begin(); layer != layers.end(); ++layer){
        (*layer)->saveToFile(path, idx);
        idx++;
    }
}

/* Refactor function */
// NeuralNetwork NeuralNetwork::fromFile(std::string path){
//     std::vector<int> row;
//     std::ifstream fin;
//     std::string line, word, temp;
//     fin.open(path + "info.txt", std::ios::in);

//     std::getline(fin, line);
//     std::stringstream s(line);
//     while(std::getline(s, word, DELIMITER)){
//         row.push_back(std::stof(word));
//     }

//     NeuralNetwork nn(row.at(0));
    
//     std::getline(fin, line);
//     int idx = 0;
//     while(std::getline(s, word, DELIMITER)){
//         nn.getLayers().at(idx)->setActivation(word);
//         idx++;
//     }

//     Matrix<float> matrix;

//     for(int i = 0; i < nn.getNumLayers() - 1; i++){
//         matrix = getMatrixFromCsv(path + std::to_string(i) + ".txt");
//         nn.getLayers().at(i)->setLayer(matrix);
//     }

//     return nn;
// }
