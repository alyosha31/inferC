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

