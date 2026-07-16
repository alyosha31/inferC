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
                     size_t length,
                     float epsilon)
{
    float output[3] = {0.0f, 0.0f, 0.0f};
    const float tolerance = 1e-5f;

    rmsnorm(weights, input, output, epsilon, length);

    for (size_t i = 0; i < length; i++) {
        assert_close(output[i], expected[i], tolerance);
    }
}

int main(void)
{
    /* RMSNorm([3, 4], [1, 2], 0): [3/sqrt(12.5), 8/sqrt(12.5)]. */
    const float weights[] = {1.0f, 2.0f};
    const float input[] = {3.0f, 4.0f};
    const float expected[] = {0.84852814f, 2.26274170f};
    run_case(weights, input, expected, 2, 0.0f);

    /* Epsilon changes the denominator and prevents division by zero. */
    const float zero_input[] = {0.0f, 0.0f, 0.0f};
    const float zero_weights[] = {1.0f, 2.0f, 3.0f};
    const float zero_expected[] = {0.0f, 0.0f, 0.0f};
    run_case(zero_weights, zero_input, zero_expected, 3, 1e-5f);

    printf("test_rmsnorm: passed\n");
    return EXIT_SUCCESS;
}
