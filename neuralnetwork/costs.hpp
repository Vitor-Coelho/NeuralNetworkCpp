#ifndef COSTS_H
#define COSTS_H

#include <cmath>
#include "matrix.hpp"


// TODO: adicionar as definições das derivadas das funções


typedef Matrix<float> (*cost_t) (Matrix<float>, bool);

inline Matrix<float> crossEntropy(Matrix<float> x, bool derivative){
    auto func = [](float x){return 1/(1 + (float)exp(-x));};
    return x.applyFunction(func);
}

inline Matrix<float> relu(Matrix<float> x, bool derivative){
    auto func = [](float x){return x > 0 ? x : 0;};
    return x.applyFunction(func);
}

inline Matrix<float> softmax(Matrix<float> x, bool derivative){
    auto func = [](float x){return (float) exp(x);};
    x = x.applyFunction(func);
    float expSum = x.sum();
    return x / expSum;
}

#endif
