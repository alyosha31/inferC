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

static void test_silu_values(void)
{
    assert_close(silu(0.0f), 0.0f, 1e-7f);
    assert_close(silu(1.0f), 0.7310586f, 1e-6f);
    assert_close(silu(-1.0f), -0.2689414f, 1e-6f);
}

static void test_swiglu_hand_calculable_case(void)
{
    const size_t dim = 2;
    const size_t hidden_dim = 3;
    const float input[] = {1.0f, 2.0f};
    const float w_gate[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    };
    const float w_up[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, -1.0f
    };
    const float w_down[] = {
        1.0f, 2.0f, 3.0f,
        -1.0f, 0.0f, 1.0f
    };
    float gate[3] = {0.0f, 0.0f, 0.0f};
    float up[3] = {0.0f, 0.0f, 0.0f};
    float output[2] = {0.0f, 0.0f};

    const float gated0 = silu(1.0f) * 1.0f;
    const float gated1 = silu(2.0f) * 2.0f;
    const float gated2 = silu(3.0f) * -1.0f;
    const float expected_output[] = {
        gated0 + 2.0f * gated1 + 3.0f * gated2,
        -gated0 + gated2
    };

    assert(swiglu(input, w_gate, w_up, w_down,
                  gate, up, output, dim, hidden_dim) == 0);

    assert_close(gate[0], gated0, 1e-6f);
    assert_close(gate[1], gated1, 1e-6f);
    assert_close(gate[2], gated2, 1e-6f);
    assert_close(up[0], 1.0f, 1e-6f);
    assert_close(up[1], 2.0f, 1e-6f);
    assert_close(up[2], -1.0f, 1e-6f);
    assert_close(output[0], expected_output[0], 1e-5f);
    assert_close(output[1], expected_output[1], 1e-5f);
}

static void test_swiglu_rejects_invalid_inputs(void)
{
    float input[1] = {1.0f};
    float weights[1] = {1.0f};
    float buffer[1] = {0.0f};

    assert(swiglu(NULL, weights, weights, weights,
                  buffer, buffer, buffer, 1, 1) == -1);
    assert(swiglu(input, weights, weights, weights,
                  buffer, buffer, buffer, 0, 1) == -1);
    assert(swiglu(input, weights, weights, weights,
                  buffer, buffer, buffer, 1, 0) == -1);
}

int main(void)
{
    test_silu_values();
    test_swiglu_hand_calculable_case();
    test_swiglu_rejects_invalid_inputs();

    printf("test_swiglu: passed\n");
    return EXIT_SUCCESS;
}
