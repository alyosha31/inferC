#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "config.h"

/* Temporary declaration until config.h exposes the public validator. */
int validate_config(ModelConfig *config);

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

static void test_valid_tiny_stories_config(void)
{
    ModelConfig config = tiny_stories_config();

    assert(validate_config(&config) == 0);
    assert(config.head_dim == 48);
    assert(config.kv_dim == 288);
    assert(config.vocab_size == 32000);
    assert(config.shared_classifier == 1);
}

static void test_negative_vocab_preserves_untied_classifier_mode(void)
{
    ModelConfig config = tiny_stories_config();
    config.raw_vocab_size = -32000;

    assert(validate_config(&config) == 0);
    assert(config.vocab_size == 32000);
    assert(config.shared_classifier == 0);
}

static void test_invalid_configurations(void)
{
    ModelConfig config = tiny_stories_config();

    assert(validate_config(NULL) == -1);

    config = tiny_stories_config();
    config.dim = 0;
    assert(validate_config(&config) == -1);

    config = tiny_stories_config();
    config.n_heads = 0;
    assert(validate_config(&config) == -1);

    config = tiny_stories_config();
    config.dim = 289;
    assert(validate_config(&config) == -1);

    config = tiny_stories_config();
    config.n_kv_heads = 7;
    assert(validate_config(&config) == -1);

    config = tiny_stories_config();
    config.raw_vocab_size = 0;
    assert(validate_config(&config) == -1);
}

int main(void)
{
    test_valid_tiny_stories_config();
    test_negative_vocab_preserves_untied_classifier_mode();
    test_invalid_configurations();

    printf("test_config: passed\n");
    return EXIT_SUCCESS;
}
