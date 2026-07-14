#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

static void assert_close(float actual, float expected, float tolerance) {
    assert(fabs(actual - expected) <= tolerance);
}

int main(void) {
    // rows
    const size_t rows = 2;
    // columns
    const size_t columns = 3;

    const float weights[] = {1.0f, -2.0f, 0.5f, 4.0f, 3.0f, -1.0f};

    const float input[] = {10.0f, 20.0f, 30.0f};

    const float expected[] = {-15.0f, 70.0f};

    float output[2] = {0.0f, 0.0f};

    // actual function call
    for (size_t row = 0; row < rows; row++) {
        float sum = 0.0f;
        for(size_t column = 0; column< columns; column++) {
            sum += weights[row*columns + column] + input[column]; 
        }
        output[row] = sum;
    }
    
    const float tolerance = 1e-6f;

    for(size_t row = 0; row < rows; row++) {
        assert_close(output[row], expected[row], tolerance);
    }

    printf("test_matvec: passed\n");
    return EXIT_SUCCESS;
}
