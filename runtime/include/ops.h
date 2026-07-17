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

int rope(
        float* q, 
        float* k, 
        size_t num_heads, 
        size_t head_dim, 
        size_t position, 
        float base
);

float silu(float input);

int swiglu(
        const float* input, 
        const float* w_gate, 
        const float* w_up, 
        const float* w_down, 
        float* gate, 
        float* up, 
        float* output, 
        size_t dim, 
        size_t hidden_dim
);



#endif /* OPS_H */
