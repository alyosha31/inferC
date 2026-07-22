#include<stdio.h>
#include<stdlib.h>
#include<stddef.h>
#include<stdint.h>

#include "checkpoint.h"
#include "config.h"

typedef struct ModelWeights {
    const float *token_embedding_table;
    const float *rms_att_weight;
    const float *wq;
    const float *wk;
    const float *wv;
    const float *wo;
    const float *rms_ffn_weight;
    const float *w_gate;
    const float *w_up;
    const float *w_down;
    const float *rms_final_weight;
    const float *wcls; // for tinystories, this is token_embedding_table
} ModelWeights;

struct InferModel {
    ModelConfig config;
    unsigned char *data;
    size_t file_size;
    ModelWeights weights;
};

enum {
    TOKEN_EMBEDDING_OFFSET = 28,
    RMS_ATT_OFFSET = 36864028,
    WQ_OFFSET = 36870940,
    WK_OFFSET = 38861596,
    WV_OFFSET = 40852252,
    WO_OFFSET = 42842908,
    RMS_FFN_OFFSET = 44833564,
    W_GATE_OFFSET = 44840476,
    W_DOWN_OFFSET = 50148892,
    W_UP_OFFSET = 55457308,
    RMS_FINAL_OFFSET = 60765724
};


static InferStatus map_weights_from_data(InferModel *model) {
    if(model == NULL) {
        return INFER_INVALID_CHECKPOINT;
    }
    if(model->data == NULL) {
        return INFER_ALLOCATION_ERROR;
    }

    ModelWeights *weights = &model->weights;

    weights->token_embedding_table = (const float *)(model->data + TOKEN_EMBEDDING_OFFSET);
    weights->rms_att_weight = (const float *)(model->data + RMS_ATT_OFFSET);
    weights->wq = (const float *)(model->data + WQ_OFFSET);
    weights->wk = (const float *)(model->data + WK_OFFSET);
    weights->wv = (const float *)(model->data + WV_OFFSET);
    weights->wo = (const float *)(model->data + WO_OFFSET);
    weights->rms_ffn_weight = (const float *)(model->data + RMS_FFN_OFFSET);
    weights->w_gate = (const float *)(model->data + W_GATE_OFFSET);
    weights->w_down = (const float *)(model->data + W_DOWN_OFFSET);
    weights->w_up = (const float *)(model->data + W_UP_OFFSET);
    weights->rms_final_weight = (const float *)(model->data + RMS_FINAL_OFFSET);
    weights->wcls = weights->token_embedding_table; // for tiny stories, they are same


    return INFER_OK;
}


int calculate_expected_size(ModelConfig config, size_t *expected_size) {
    // validation
    int validate = validate_config(&config);
    if(validate != 0) {
        return -1;
    }

    size_t total = 0;
    total += config.vocab_size*config.dim; // embedding
    total += config.n_layers*config.dim; // rms_att
    total += config.n_layers*config.dim*config.dim; // Wq
    total += config.n_layers*config.dim*config.kv_dim; // Wk
    total += config.n_layers*config.dim*config.kv_dim; // Wv
    total += config.n_layers*config.dim*config.dim; // Wo
    total += config.n_layers*config.dim; // rms_ffn 
    total += config.n_layers*config.hidden_dim*config.dim; // Wgate
    total += config.n_layers*config.dim*config.hidden_dim; // Wdown
    total += config.n_layers*config.hidden_dim*config.dim; // Wup 
    total += config.dim; // rms_final
    total += config.seq_len*config.head_dim;

    if(config.raw_vocab_size < 0) {
        total += config.vocab_size*config.dim; // W output if Woutput != E
    }

    size_t weight_bytes = total * sizeof(float);
    size_t header_bytes = 7 * sizeof(int32_t);

    *expected_size = weight_bytes + header_bytes;
    return 0;
}

InferStatus model_load(const char *path, InferModel **out_model) {
    // argument validation
    if(path == NULL || out_model == NULL) return INFER_INVALID_ARGUMENT;
    *out_model = NULL;

    // file opening
    FILE *file = fopen(path, "rb");
    if(file == NULL) {
        return INFER_IO_ERROR;
    }

    // header reading
    int32_t raw_header[7]; // tinystories checkpoint header is 7 integers (28 bytes)
    size_t fields_read = fread(
        raw_header,
        sizeof(raw_header[0]),
        7,
        file
    );
    
    if(fields_read != 7) {
        fclose(file);
        return INFER_INVALID_CHECKPOINT;
    }

    // configuration validation
    ModelConfig config = {0};
    if(
        raw_header[0] <= 0 ||
        raw_header[1] <= 0 ||
        raw_header[2] <= 0 ||
        raw_header[3] <= 0 ||
        raw_header[4] <= 0 ||
        raw_header[6] <= 0
    ) {
        fclose(file);
        return INFER_INVALID_CHECKPOINT;
    }

    config.dim = (size_t) raw_header[0];
    config.hidden_dim = (size_t) raw_header[1];
    config.n_layers = (size_t) raw_header[2];
    config.n_heads = (size_t) raw_header[3];
    config.n_kv_heads = (size_t) raw_header[4];
    config.raw_vocab_size = raw_header[5]; 
    config.seq_len = (size_t) raw_header[6];

    if(validate_config(&config) != 0) {
        fclose(file);
        return INFER_UNSUPPORTED_CONFIG;
    }

    if(fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return INFER_IO_ERROR;
    }

    long end_offset = ftell(file);
    if(end_offset < 0) {
        fclose(file);
        return INFER_IO_ERROR;
    }
    size_t file_size = (size_t) end_offset;
    size_t expected_size;
    int calculate_expected_res = calculate_expected_size(config, &expected_size);
    if(calculate_expected_res != 0) {
        fclose(file);
        return INFER_IO_ERROR;
    }
    if(expected_size != file_size) {
        fclose(file);
        return INFER_INVALID_CHECKPOINT;
    }
    if(fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return INFER_IO_ERROR;
    }
    // allocation
    InferModel *model = malloc(sizeof(*model));
    if(model == NULL) {
        fclose(file);
        return INFER_ALLOCATION_ERROR;
    }
    // initialization
    model->config = config;
    model->file_size = file_size;
    model->data = malloc(file_size); // no sizeof since file_size is in binary
    if(model->data == NULL) {
        free(model);
        fclose(file);
        return INFER_ALLOCATION_ERROR;
    }

    size_t bytes_read = fread(model->data, 1, file_size, file);
    if(bytes_read != file_size) {
        free(model->data);
        free(model);
        fclose(file);
        return INFER_IO_ERROR;
    }

    // weight mapping
    InferStatus map_weights_status = map_weights_from_data(model);
    if(map_weights_status != INFER_OK) {
        free(model->data);
        free(model);
        fclose(file);
        return map_weights_status;
    }

    fclose(file);
    *out_model = model;

    return INFER_OK;
}


void model_destroy(InferModel *model) {
    if(model == NULL) {
        return;
    }

    free(model->data);
    free(model);
}

