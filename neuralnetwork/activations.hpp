#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <cmath>
#include "matrix.hpp"


// TODO: adicionar as definições das derivadas das funções


typedef Matrix<float> (*activation_t) (Matrix<float>, bool);

inline Matrix<float> sigmoid(Matrix<float> x, bool derivative){
    auto func = [](float x){return 1/(1 + (float)exp(-x));};
    if(!derivative){
        return x.applyFunction(func);
    }else{
        Matrix<float> sig = x.applyFunction(func);
        return sig; // sig.multiplyElemWise(sig*(-1) + 1);
    }
}

inline Matrix<float> relu(Matrix<float> x, bool derivative){
    if(!derivative){
        auto func = [](float x){return x > 0 ? x : 0;};
        return x.applyFunction(func);
    }else{
        auto func = [](float x){return x > 0 ? (float) 1 : 0;};
        return x.applyFunction(func);
    }
}

inline Matrix<float> softmax(Matrix<float> x, bool derivative){
    auto func = [](float x){return (float) exp(x);};
    x = x.applyFunction(func);
    float expSum = x.sum();
    return x / expSum;
}

#endif
