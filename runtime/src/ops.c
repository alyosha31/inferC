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

int rope(float* q, float* k, size_t num_heads, size_t head_dim, size_t position, float base) {
    if(q == NULL || k == NULL || num_heads == 0 || head_dim == 0 || head_dim%2 || base == 0) {
        return -1;
    }

    // since q and k are flattened, we must calculate for each head we are working with
    for(size_t head = 0; head < num_heads; head++) {
        size_t index = head_dim*head; // starting index
        for(size_t i = 0; i < head_dim/2; i++) {
            size_t ind = index + 2*i; // index of the first element in the pair
            float theta = (float)position*powf((float)base, (-(2.0f)*(i)/((float)head_dim)));
            float prev_first_query = q[ind];
            float prev_second_query = q[ind+1];
            float prev_first_key = k[ind];
            float prev_second_key = k[ind+1];
            float cosine = cosf(theta);
            float sine = sinf(theta); 
            q[ind] = prev_first_query*cosine - prev_second_query*sine;
            q[ind+1] = prev_first_query*sine + prev_second_query*cosine;
            k[ind] = prev_first_key*cosine - prev_second_key*sine;
            k[ind+1] = prev_first_key*sine + prev_second_key*cosine;
        }
    }

    return 0;
}
