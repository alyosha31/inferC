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


#endif /* OPS_H */
