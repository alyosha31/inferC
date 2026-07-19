#include<stdio.h>
#include<stdlib.h>
#include<stddef.h>

#include "config.h"


int validate_config(ModelConfig *config) {
    if(config == NULL) {
        return -1;
    }
    if(
        config->dim == 0 || 
        config->hidden_dim == 0 || 
        config->n_layers == 0 ||
        config->n_heads == 0 ||
        config->n_kv_heads == 0 || 
        config->raw_vocab_size == 0 ||
        config->seq_len == 0 || 
        config->dim%config->n_heads != 0 || 
        config->n_kv_heads > config->n_heads
    ) {
        return -1;
    }
    config->head_dim = dim/n_heads;
    config->kv_dim = dim/n_kv_heads;
    config->shared_classifier = (raw_vocab_size > 0)?1:0; // for tinystories, E == W_vocab so this should be 1
    config->vocab_size = (size_t) abs(raw_vocab_size);
    
    return 0;
}


