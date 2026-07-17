#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "ops.h"

/* Temporary test declaration until rope is added to the public ops header. */
int rope(float *q, float *k, size_t num_heads, size_t head_dim,
         size_t position, float base);

static void assert_close(float actual, float expected, float tolerance)
{
    assert(fabsf(actual - expected) <= tolerance);
}

static void assert_array_close(const float *actual,
                               const float *expected,
                               size_t length,
                               float tolerance)
{
    for (size_t i = 0; i < length; i++) {
        assert_close(actual[i], expected[i], tolerance);
    }
}

static void test_position_zero_is_identity(void)
{
    const float original_q[] = {1.0f, -2.0f, 3.0f, -4.0f,
                                5.0f, -6.0f, 7.0f, -8.0f};
    const float original_k[] = {-8.0f, 7.0f, -6.0f, 5.0f,
                                -4.0f, 3.0f, -2.0f, 1.0f};
    float q[8];
    float k[8];

    for (size_t i = 0; i < 8; i++) {
        q[i] = original_q[i];
        k[i] = original_k[i];
    }

    assert(rope(q, k, 2, 4, 0, 10000.0f) == 0);
    assert_array_close(q, original_q, 8, 0.0f);
    assert_array_close(k, original_k, 8, 0.0f);
}

static void test_known_rotation_for_one_head(void)
{
    const float tolerance = 1e-5f;
    const float cosine_fast = cosf(1.0f);
    const float sine_fast = sinf(1.0f);
    const float cosine_slow = cosf(0.01f);
    const float sine_slow = sinf(0.01f);
    float q[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float k[] = {0.0f, 1.0f, 1.0f, 0.0f};
    const float expected_q[] = {cosine_fast, sine_fast,
                                -sine_slow, cosine_slow};
    const float expected_k[] = {-sine_fast, cosine_fast,
                                cosine_slow, sine_slow};

    assert(rope(q, k, 1, 4, 1, 10000.0f) == 0);
    assert_array_close(q, expected_q, 4, tolerance);
    assert_array_close(k, expected_k, 4, tolerance);
}

static void test_heads_do_not_cross_boundaries(void)
{
    const float tolerance = 1e-5f;
    float q[] = {1.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 1.0f, 0.0f};
    float k[] = {0.0f, 1.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 1.0f};

    assert(rope(q, k, 2, 4, 1, 10000.0f) == 0);

    assert_close(q[0], cosf(1.0f), tolerance);
    assert_close(q[1], sinf(1.0f), tolerance);
    assert_close(q[4], 0.0f, tolerance);
    assert_close(q[5], 0.0f, tolerance);
    assert_close(q[6], cosf(0.01f), tolerance);
    assert_close(q[7], sinf(0.01f), tolerance);
}

static void test_rotation_preserves_pair_norms(void)
{
    const float before_q[] = {3.0f, 4.0f, -2.0f, 5.0f};
    const float before_k[] = {1.0f, -7.0f, 6.0f, 8.0f};
    float q[] = {3.0f, 4.0f, -2.0f, 5.0f};
    float k[] = {1.0f, -7.0f, 6.0f, 8.0f};

    assert(rope(q, k, 1, 4, 17, 10000.0f) == 0);

    for (size_t i = 0; i < 4; i += 2) {
        const float q_before_norm = before_q[i] * before_q[i]
                                  + before_q[i + 1] * before_q[i + 1];
        const float q_after_norm = q[i] * q[i] + q[i + 1] * q[i + 1];
        const float k_before_norm = before_k[i] * before_k[i]
                                  + before_k[i + 1] * before_k[i + 1];
        const float k_after_norm = k[i] * k[i] + k[i + 1] * k[i + 1];

        assert_close(q_after_norm, q_before_norm, 1e-4f);
        assert_close(k_after_norm, k_before_norm, 1e-4f);
    }
}

static void test_invalid_inputs(void)
{
    float q[4] = {0.0f};
    float k[4] = {0.0f};

    assert(rope(NULL, k, 1, 4, 0, 10000.0f) == -1);
    assert(rope(q, NULL, 1, 4, 0, 10000.0f) == -1);
    assert(rope(q, k, 0, 4, 0, 10000.0f) == -1);
    assert(rope(q, k, 1, 0, 0, 10000.0f) == -1);
    assert(rope(q, k, 1, 3, 0, 10000.0f) == -1);
    assert(rope(q, k, 1, 4, 0, 0.0f) == -1);
}

int main(void)
{
    test_position_zero_is_identity();
    test_known_rotation_for_one_head();
    test_heads_do_not_cross_boundaries();
    test_rotation_preserves_pair_norms();
    test_invalid_inputs();

    printf("test_rope: passed\n");
    return EXIT_SUCCESS;
}
