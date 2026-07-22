CC := cc
CPPFLAGS := -Iinclude -MMD -MP
SANITIZERS := -fsanitize=address,undefined
CFLAGS := -std=c17 -Wall -Wextra -Wpedantic -Werror -g $(SANITIZERS)
LDFLAGS := $(SANITIZERS)
LDLIBS := -lm

BUILD_DIR := build
TEST_BINS := $(BUILD_DIR)/test_matvec $(BUILD_DIR)/test_rmsnorm $(BUILD_DIR)/test_softmax $(BUILD_DIR)/test_rope $(BUILD_DIR)/test_swiglu $(BUILD_DIR)/test_config $(BUILD_DIR)/test_checkpoint $(BUILD_DIR)/test_state
RUNTIME_SRC := src/ops.c
RUNTIME_OBJ := $(BUILD_DIR)/$(RUNTIME_SRC:.c=.o)
CONFIG_SRC := src/config.c
CONFIG_OBJ := $(BUILD_DIR)/$(CONFIG_SRC:.c=.o)
CHECKPOINT_SRC := src/checkpoint.c
CHECKPOINT_OBJ := $(BUILD_DIR)/$(CHECKPOINT_SRC:.c=.o)
STATE_SRC := src/state.c
STATE_OBJ := $(BUILD_DIR)/$(STATE_SRC:.c=.o)
TEST_SRCS := tests/test_matvec.c tests/test_rmsnorm.c tests/test_softmax.c tests/test_rope.c tests/test_swiglu.c tests/test_config.c tests/test_checkpoint.c tests/test_state.c
TEST_OBJS := $(TEST_SRCS:%.c=$(BUILD_DIR)/%.o)
TEST_DEPS := $(TEST_OBJS:.o=.d)
RUNTIME_DEPS := $(RUNTIME_OBJ:.o=.d)

.PHONY: all test clean

all: test

$(BUILD_DIR)/test_matvec: $(BUILD_DIR)/tests/test_matvec.o $(RUNTIME_OBJ) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/test_rmsnorm: $(BUILD_DIR)/tests/test_rmsnorm.o $(RUNTIME_OBJ) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/test_softmax: $(BUILD_DIR)/tests/test_softmax.o $(RUNTIME_OBJ) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/test_rope: $(BUILD_DIR)/tests/test_rope.o $(RUNTIME_OBJ) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/test_swiglu: $(BUILD_DIR)/tests/test_swiglu.o $(RUNTIME_OBJ) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/test_config: $(BUILD_DIR)/tests/test_config.o $(CONFIG_OBJ) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/test_checkpoint: $(BUILD_DIR)/tests/test_checkpoint.o $(CHECKPOINT_OBJ) $(CONFIG_OBJ) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/test_state: $(BUILD_DIR)/tests/test_state.o $(STATE_OBJ) $(CONFIG_OBJ) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

$(RUNTIME_OBJ): $(RUNTIME_SRC) include/ops.h

$(CONFIG_OBJ): $(CONFIG_SRC) include/config.h

$(CHECKPOINT_OBJ): $(CHECKPOINT_SRC) include/checkpoint.h include/config.h

$(STATE_OBJ): $(STATE_SRC) include/state.h include/config.h

-include $(TEST_DEPS) $(RUNTIME_DEPS)

test: $(TEST_BINS)
	@for test in $(TEST_BINS); do \
		$$test; \
	done

clean:
	rm -rf $(BUILD_DIR)
