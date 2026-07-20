#ifndef CHECKPOINT_H
#define CHECKPOINT_H

typedef struct InferModel InferModel;

typedef enum {
    INFER_OK = 0,
    INFER_INVALID_ARGUMENT = -1,
    INFER_IO_ERROR = -2,
    INFER_INVALID_CHECKPOINT = -3,
    INFER_UNSUPPORTED_CONFIG = -4,
    INFER_ALLOCATION_ERROR = -5
} InferStatus;

InferStatus model_load(const char *path, InferModel **out_model);
void model_destroy(InferModel *model);

#endif /* CHECKPOINT_H */
