CC := cc
CPPFLAGS := -Iinclude -MMD -MP
CFLAGS := -std=c17 -Wall -Wextra -Wpedantic -Werror -g
SANITIZERS := -fsanitize=address,undefined
LDFLAGS := $(SANITIZERS)
LDLIBS := -lm

BUILD_DIR := build
TEST_BINS := $(BUILD_DIR)/test_matvec $(BUILD_DIR)/test_rmsnorm
RUNTIME_SRC := src/ops.c
RUNTIME_OBJ := $(BUILD_DIR)/$(RUNTIME_SRC:.c=.o)
TEST_SRCS := tests/test_matvec.c tests/test_rmsnorm.c
TEST_OBJS := $(TEST_SRCS:%.c=$(BUILD_DIR)/%.o)
TEST_DEPS := $(TEST_OBJS:.o=.d)
RUNTIME_DEPS := $(RUNTIME_OBJ:.o=.d)

.PHONY: all test clean

all: test

$(BUILD_DIR)/test_matvec: $(BUILD_DIR)/tests/test_matvec.o $(RUNTIME_OBJ) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/test_rmsnorm: $(BUILD_DIR)/tests/test_rmsnorm.o $(RUNTIME_OBJ) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

$(RUNTIME_OBJ): $(RUNTIME_SRC) include/ops.h

-include $(TEST_DEPS) $(RUNTIME_DEPS)

test: $(TEST_BINS)
	@for test in $(TEST_BINS); do \
		$$test; \
	done

clean:
	rm -rf $(BUILD_DIR)
