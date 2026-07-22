#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "state.h"

static ModelConfig tiny_stories_config(void)
{
    ModelConfig config = {0};
    config.dim = 288;
    config.hidden_dim = 768;
    config.n_layers = 6;
    config.n_heads = 6;
    config.n_kv_heads = 6;
    config.raw_vocab_size = 32000;
    config.seq_len = 256;
    return config;
}

static void test_invalid_output_pointer(void)
{
    assert(state_create(tiny_stories_config(), NULL) == -1);
}

static void test_invalid_configuration(void)
{
    ModelConfig config = tiny_stories_config();
    RuntimeState *state = NULL;

    config.dim = 0;

    assert(state_create(config, &state) == -1);
    assert(state == NULL);
}

static void test_create_and_destroy(void)
{
    RuntimeState *state = NULL;

    assert(state_create(tiny_stories_config(), &state) == 0);
    assert(state != NULL);
    state_destroy(state);
}

int main(void)
{
    test_invalid_output_pointer();
    test_invalid_configuration();
    test_create_and_destroy();
    state_destroy(NULL);

    printf("test_state: passed\n");
    return EXIT_SUCCESS;
}
