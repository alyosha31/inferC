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
    if(rows == 0) return -1; // division by zero error

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
