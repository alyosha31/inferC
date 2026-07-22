#ifndef STATE_H
#define STATE_H
#include<stddef.h>
#include "config.h"


typedef struct RuntimeState RuntimeState;

int state_create(ModelConfig config, RuntimeState **out_state);
void state_destroy(RuntimeState *state);

#endif
