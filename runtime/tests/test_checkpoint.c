#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "checkpoint.h"

static const char *invalid_checkpoint_path = "build/test_invalid_checkpoint.bin";
static const char *tiny_stories_path = "../models/stories15M.bin";

static void write_header_only_checkpoint(void)
{
    const int32_t header[7] = {
        288,
        768,
        6,
        6,
        6,
        32000,
        256
    };

    FILE *file = fopen(invalid_checkpoint_path, "wb");
    assert(file != NULL);
    assert(fwrite(header, sizeof(header[0]), 7, file) == 7);
    assert(fclose(file) == 0);
}

static void test_invalid_arguments(void)
{
    InferModel *model = NULL;

    assert(model_load(NULL, &model) == INFER_INVALID_ARGUMENT);
    assert(model == NULL);
    assert(model_load(tiny_stories_path, NULL) == INFER_INVALID_ARGUMENT);
}

static void test_missing_checkpoint(void)
{
    InferModel *model = NULL;

    assert(model_load("build/does-not-exist.bin", &model) == INFER_IO_ERROR);
    assert(model == NULL);
}

static void test_truncated_checkpoint(void)
{
    InferModel *model = NULL;

    write_header_only_checkpoint();
    assert(model_load(invalid_checkpoint_path, &model) == INFER_INVALID_CHECKPOINT);
    assert(model == NULL);
    assert(remove(invalid_checkpoint_path) == 0);
}

static void test_tiny_stories_checkpoint(void)
{
    InferModel *model = NULL;

    assert(model_load(tiny_stories_path, &model) == INFER_OK);
    assert(model != NULL);
    model_destroy(model);
}

int main(void)
{
    test_invalid_arguments();
    test_missing_checkpoint();
    test_truncated_checkpoint();
    test_tiny_stories_checkpoint();
    model_destroy(NULL);

    printf("test_checkpoint: passed\n");
    return EXIT_SUCCESS;
}
