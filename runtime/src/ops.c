#include <math.h>
#include<stdio.h>
#include<stdlib.h>
#include<stddef.h>
#include "ops.h"

void matvec(const float *weights, const float *input, float *output, size_t rows, size_t columns) {
    for(size_t row = 0; row<rows; row++) {
        float sum = 0; 
        for(size_t col = 0; col<columns; col++) {
            sum += weights[row*columns + col]*input[col];
        }
        output[row] = sum;
    }    
}

int rmsnorm(const float* weights, const float *input, float* output, const float epsilon, size_t rows) {
    // RMSNorm scales each feature after normalizing the input magnitude.
    if(rows == 0 || weights == NULL || input == NULL || output == NULL || epsilon <= 0) return -1; // invalid input 

    float meanSquare = 0.0f;
    for(size_t i = 0; i < rows; i++) {
        meanSquare+= input[i]*input[i];
    }
    meanSquare = meanSquare / rows;
    float divisor = sqrtf(meanSquare + epsilon);
    float scale = 1/(divisor);
    for(size_t i = 0; i < rows; i++) {
        output[i] = weights[i]*input[i]*scale;
    }

    return 0;
}

int softmax(const float* input, float *output, size_t length) {
    // validating input
    if(input == NULL || output == NULL || length == 0) return -1;
    // finding the maximum of input values to exponential becomes safe 
    float maxInput = input[0];
    for(size_t i = 0; i < length; i++) {
        if(input[i] > maxInput) {
            maxInput = input[i];
        } 
    }

    float sum = 0;
    for(size_t i = 0; i < length; i++) {
        output[i] = expf(input[i] - maxInput);
        sum += output[i];
    }

    for(size_t i = 0; i < length; i++) {
        output[i] /= sum;
    }

    return 0;
}
