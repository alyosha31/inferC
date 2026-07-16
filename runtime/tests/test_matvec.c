#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "ops.h"

static void assert_close(float actual, float expected, float tolerance)
{
    assert(fabsf(actual - expected) <= tolerance);
}

static void run_case(const float *weights,
                     const float *input,
                     const float *expected,
                     size_t rows,
                     size_t columns)
{
    float output[2] = {0.0f, 0.0f};
    const float tolerance = 1e-6f;

    matvec(weights, input, output, rows, columns);

    for (size_t row = 0; row < rows; row++) {
        assert_close(output[row], expected[row], tolerance);
    }
}

int main(void)
{
    const float weights[] = {
        1.0f, -2.0f, 0.5f,
        4.0f, 3.0f, -1.0f
    };
    const float input[] = {10.0f, 20.0f, 30.0f};
    const float expected[] = {-15.0f, 70.0f};
    run_case(weights, input, expected, 2, 3);

    const float one_weight[] = {3.0f};
    const float one_input[] = {4.0f};
    const float one_expected[] = {12.0f};
    run_case(one_weight, one_input, one_expected, 1, 1);

    const float zero_weights[] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float zero_input[] = {10.0f, -5.0f};
    const float zero_expected[] = {0.0f, 0.0f};
    run_case(zero_weights, zero_input, zero_expected, 2, 2);

    printf("test_matvec: passed\n");
    return EXIT_SUCCESS;
}
