#include<stdio.h>
#include<stdlib.h>
#include<stddef.h>
#include "state.h"
#include "config.h"

struct RuntimeState {
    ModelConfig config;
    // x -> current residual stream
    float *x;
    // xb -> normalized state
    float *xb;
    // xb2 -> after Wo
    float *xb2;
    // q -> query projection
    float *q;
    // k -> key projection
    float *k;
    // v -> value projection
    float *v;
    // att -> attention scores
    float *att;
    // gate -> swiglu gate scratch
    float *gate;
    // up -> swiglu up scratch
    float *up;
    // logits -> next token scores
    float *logits;

    // key_cache [n_layers][seq_len][kv_dim];
    float *key_cache;
    // value_cache [n_layers][seq_len][kv_dim];
    float *value_cache;
};

int state_create(ModelConfig config, RuntimeState **out_state) {
    if(out_state == NULL) {
        return -1;
    }
    *out_state = NULL;
    if(validate_config(&config) != 0) {
        return -1;
    }

    RuntimeState* state = calloc(1, sizeof(*state));
    if(state == NULL) {
        return -2; // failed allocation
    }
    state->config = config;
    // allocations
    // x
    state->x = malloc(config.dim*sizeof(*state->x));
    if(state->x == NULL) {
        state_destroy(state);
        return -2; // failed allocation
    }
    state->xb = malloc(config.dim*sizeof(*state->xb));
    if(state->xb == NULL) {
        state_destroy(state);
        return -2; // failed allocation
    }
    state->xb2 = malloc(config.dim * sizeof(*state->xb2));
    if(state->xb2 == NULL) {
        state_destroy(state);
        return -2; // failed allocation
    }
    state->q = malloc(config.dim * sizeof(*state->q));
    if(state->q == NULL) {
        state_destroy(state);
        return -2; // failed allocation
    }
    state->k = malloc(config.kv_dim * sizeof(*state->k));
    if(state->k == NULL) {
        state_destroy(state);
        return -2; // failed allocation
    }
    state->v = malloc(config.kv_dim*sizeof(*state->v));
    if(state->v == NULL) {
        state_destroy(state);
        return -2; // failed allocation
    }
    state->logits = malloc(config.vocab_size*sizeof(*state->logits));
    if(state->logits == NULL) {
        state_destroy(state);
        return -2; // failed allocation
    }
    state->att = malloc(config.n_heads*config.seq_len*sizeof(*state->att));
    if(state->att == NULL) {
        state_destroy(state);
        return -2; // failed allocation
    }
    state->gate = malloc(config.hidden_dim*sizeof(*state->gate));
    if(state->gate == NULL) {
        state_destroy(state);
        return -2; // failed allocation
    }
    state->up = malloc(config.hidden_dim*sizeof(*state->up));
    if(state->up == NULL) {
        state_destroy(state);
        return -2; // failed allocation
    }
    // kv cache allocation n_layers*seq_len*kv_dim
    state->key_cache = malloc(config.n_layers*config.seq_len*config.kv_dim*sizeof(*state->key_cache));
    if(state->key_cache == NULL) {
        state_destroy(state);
        return -2;
    }

    state->value_cache = malloc(config.n_layers*config.seq_len*config.kv_dim*sizeof(*state->value_cache));
    if(state->value_cache == NULL) {
        state_destroy(state);
        return -2;
    }

    // assign state to the value of the pointer that out_state points to.
    *out_state = state;
    return 0;
}

void state_destroy(RuntimeState *state) {
    if(state == NULL) {
        return;
    }
    free(state->x);
    free(state->xb);
    free(state->xb2);
    free(state->q);
    free(state->k);
    free(state->v);
    free(state->logits);
    free(state->att);
    free(state->gate);
    free(state->up);
    free(state->key_cache);
    free(state->value_cache);
    free(state);
}
