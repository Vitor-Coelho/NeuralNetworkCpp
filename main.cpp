#include <iostream>
#include "include/matrix.hpp"
#include "include/nn.hpp"

using namespace std;

#define NUM_LAYERS  4
#define LAYERS      5,10,6,4
#define ACTIVATIONS sigmoid,relu,softmax


int main(){
    NeuralNetwork nn(NUM_LAYERS, {LAYERS}, {ACTIVATIONS});

    nn.printWeights();

    Matrix output = nn.feedforward(Matrix(1,5,"random"));

    output.print();

    return 0;
}
