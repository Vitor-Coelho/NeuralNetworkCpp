#ifndef NN_H
#define NN_H

#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include "matrix.hpp"
#include "activations.hpp"
#include "layers.hpp"
#include "costs.hpp"
#include "dataset.hpp"


// TODO
// • método de salvar layer para arquivo recebe o path origem + número para escrever num_info.txt e num.txt - informações e dados (ex.: 2_info.txt e 2.txt)
// • criar um método para concatenar redes -> por exemplo uma convolucional com uma fully connected (só garantir tamanhos corretos na interface)
// • revisar includes
// • rever getInputSize e OutputSize da FCLayer (utilidade e como melhorar)
// • fazer o mesmo que é feito para activation functions nas cost functions
// • comentar e organizar os códigos
// • nn.print(bool detailed) -> se detailed for false, apenas as informações gerais das camadas são printadas
//   se for true, printa com os pesos etc (layer.print também receberá bool detailed)

class NeuralNetwork{

    private:
        size_t numLayers, inputSize, outputSize;
        std::vector<Layer*> layers;

    public:
        NeuralNetwork(size_t numberOfLayers);
        NeuralNetwork(size_t numberOfLayers, std::vector<Layer*> NNLayers);
        ~NeuralNetwork();

        void printInfo();

        int getNumLayers();
        size_t getInputSize();
        size_t getOutputSize();
        std::vector<Layer*> getLayers();

        void assignNN(NeuralNetwork nn);
        void operator=(NeuralNetwork nn);

        Matrix<float> feedforward(Matrix<float> input);
        void train(Dataset dataset, int epochs, float learningRate, cost_t costFunc, cost_deriv_t costDer);
        float trainBatch(Matrix<float> input, Matrix<float> target, float learningRate, cost_t costFunc, cost_deriv_t costDer);
        void saveToFile(std::string path);
        static NeuralNetwork fromFile(std::string path);
};

#endif
