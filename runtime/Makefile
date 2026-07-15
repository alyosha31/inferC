CC := cc
CPPFLAGS := -Iinclude -MMD -MP
CFLAGS := -std=c17 -Wall -Wextra -Wpedantic -Werror -g
SANITIZERS := -fsanitize=address,undefined
LDFLAGS := $(SANITIZERS)
LDLIBS := -lm

BUILD_DIR := build
TEST_BIN := $(BUILD_DIR)/test_matvec
SOURCES := tests/test_matvec.c src/ops.c
OBJECTS := $(SOURCES:%.c=$(BUILD_DIR)/%.o)
DEPS := $(OBJECTS:.o=.d)

.PHONY: all test clean

all: test

$(TEST_BIN): $(OBJECTS) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $(OBJECTS) $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

-include $(DEPS)

test: $(TEST_BIN)
	$(TEST_BIN)

clean:
	rm -rf $(BUILD_DIR)
