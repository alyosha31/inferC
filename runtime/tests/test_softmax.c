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

int main(void)
{
    const float logits[] = {1000.0f, 1001.0f};
    const float expected[] = {0.26894143f, 0.73105860f};
    float probabilities[2] = {0.0f, 0.0f};

    assert(softmax(logits, probabilities, 2) == 0);
    assert_close(probabilities[0], expected[0], 1e-5f);
    assert_close(probabilities[1], expected[1], 1e-5f);
    assert_close(probabilities[0] + probabilities[1], 1.0f, 1e-6f);

    assert(softmax(NULL, probabilities, 2) == -1);
    assert(softmax(logits, probabilities, 0) == -1);

    printf("test_softmax: passed\n");
    return EXIT_SUCCESS;
}
