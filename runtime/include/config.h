#ifndef CONFIG_H
#define CONFIG_H
#include<stddef.h>
#include <stdint.h>


typedef struct {
    size_t dim;
    size_t hidden_dim;
    size_t n_layers;
    size_t n_heads;
    size_t n_kv_heads;
    int32_t raw_vocab_size;
    
    size_t seq_len;
    
    size_t head_dim;
    size_t kv_dim;
    size_t vocab_size;
    int shared_classifier;
} ModelConfig;



int validate_config(ModelConfig *config);

#endif /* CONFIG_H */
