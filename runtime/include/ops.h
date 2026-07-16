#ifndef OPS_H
#define OPS_H

#include<stddef.h>

void matvec(
        const float *weights, 
        const float *input, 
        float *output, 
        size_t rows, 
        size_t columns
);

// return -1 for division by zero errors
int rmsnorm(
        const float *weights,
        const float *input,
        float *output,
        float epsilon,
        size_t rows
);

int softmax(
        const float* input, 
        float *output, 
        size_t length
);

#endif /* OPS_H */
