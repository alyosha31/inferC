# Configuration and checkpoint-loading design

## Goal

Turn the pinned TinyStories 15M binary checkpoint into a validated, immutable model object whose weight regions can later be consumed by the forward pass.

This phase does not allocate activations, implement the KV cache, tokenize text, or execute attention. It ends at safe model loading and destruction.

## Scope

### Included

- represent the seven checkpoint header fields;
- derive and validate head_dim and kv_dim;
- read the fixed little-endian header;
- validate file size and every serialized weight region;
- allocate one model-owned FP32 checkpoint buffer;
- expose typed, read-only views of weight regions internally;
- represent tied classifier weights;
- handle malformed files and allocation failures;
- release all model-owned resources.

### Excluded

- memory mapping;
- alternate checkpoint formats;
- quantized or non-FP32 weights;
- GQA-specific behavior beyond representing n_kv_heads;
- tokenizer loading;
- activation/session allocation;
- transformer execution;
- public inference API.

## Requirements

A successful load must prove:

1. the file can be opened and read;
2. the header contains valid positive dimensions;
3. dim % n_heads == 0;
4. n_kv_heads <= n_heads;
5. the current MVP's n_heads == n_kv_heads;
6. all size calculations avoid integer overflow;
7. the calculated final byte offset equals the actual file size;
8. all weight pointers remain inside the owned checkpoint buffer;
9. positive vocab_size creates a tied classifier view;
10. destruction is safe after every partial failure.

## Files to create

### Public headers

runtime/include/config.h

Defines the stable configuration value type and validation/derivation functions. It must not contain checkpoint file handles or weight pointers.

runtime/include/checkpoint.h

Declares an opaque checkpoint/model type and the load/destroy operations. It should expose only stable caller-facing status values and read-only metadata accessors needed later.

runtime/include/model.h

Defines internal model-facing weight-view types only if a separate model module is useful. Prefer keeping serialized-pointer details private until the forward-pass module exists.

### Private sources

runtime/src/config.c

Validates raw header values and derives:

~~~text
head_dim = dim / n_heads
kv_dim   = n_kv_heads * head_dim
~~~

runtime/src/checkpoint.c

Owns file I/O, header decoding, checked size arithmetic, allocation, pointer assignment, and cleanup.

The full weight-view struct should live here or in a private header under runtime/src/; it should not be exposed through a public include.

### Tests

runtime/tests/test_config.c

Tests valid configuration, invalid relationships, and derived dimensions.

runtime/tests/test_checkpoint.c

Uses a small generated fixture rather than the 58 MB production checkpoint for most cases. Tests valid loading, truncation, invalid header values, tied classifier behavior, and destruction.

runtime/tests/test_checkpoint_real.c

Optional integration test that opens ../models/stories15M.bin, verifies the pinned configuration and file size, and checks representative offsets. This test may be skipped when model artifacts are unavailable.

### Python tools

tools/inspect_checkpoint.py

Reads the header, calculates expected regions, compares expected and actual file size, and prints a machine-readable summary. It is an oracle/inspection tool, not production runtime code.

tools/make_checkpoint_fixture.py

Generates a tiny deterministic checkpoint with the same layout and a configurable header. This is permitted fixture-generation code and makes truncation/error tests fast.

### Documentation

Already completed:

- docs/model-contract.md

This phase should additionally update it only if implementation evidence reveals a discrepancy.

## Configuration contract

Logical fields:

~~~text
dim
hidden_dim
n_layers
n_heads
n_kv_heads
vocab_size
seq_len
~~~

Derived fields:

~~~text
head_dim = dim / n_heads
kv_dim   = n_kv_heads * head_dim
shared_classifier = (raw_vocab_size > 0)
~~~

The signed vocabulary field is a format convention:

~~~text
positive vocab_size -> classifier reuses token embeddings
negative vocab_size -> classifier has a separate serialized matrix
~~~

The MVP accepts only the positive/tied case, but the loader should detect the sign before taking the absolute vocabulary size.

## Weight-view design

The checkpoint stores all FP32 values contiguously. Internally, the model needs a view containing pointers and no duplicate weight data:

~~~text
token_embedding       [vocab_size, dim]
rms_att                [n_layers, dim]
wq                     [n_layers, dim, dim]
wk                     [n_layers, kv_dim, dim]
wv                     [n_layers, kv_dim, dim]
wo                     [n_layers, dim, dim]
rms_ffn                [n_layers, dim]
w_gate                 [n_layers, hidden_dim, dim]
w_down                 [n_layers, dim, hidden_dim]
w_up                   [n_layers, hidden_dim, dim]
rms_final              [dim]
w_vocab                token_embedding when tied
~~~

Weights are immutable after loading. A pointer into the checkpoint buffer is valid only while the model owns that buffer.

For layer l, the first row of a matrix region is addressed conceptually by:

~~~text
layer_base + l * rows_per_layer * columns
~~~

The checkpoint loader records byte offsets; numerical operations later receive the corresponding flat FP32 pointer and explicit dimensions.

## Ownership

### Model owns

- copied configuration;
- checkpoint byte buffer;
- internal weight-view pointers;
- derived metadata.

### Caller owns

- the path string passed to load;
- any output metadata copied out through accessors.

### Model does not own

- activation buffers;
- KV caches;
- tokenizer state;
- caller-owned strings.

Destroying the model releases the byte buffer exactly once. A failed load must release the file handle and any allocated buffer before returning.

## Loading sequence

Conceptual sequence:

~~~text
validate arguments
open file
read exactly 28-byte header
decode little-endian signed integers
validate and derive configuration
seek/read actual file size
calculate expected size with checked arithmetic
reject mismatch
allocate model-owned byte buffer
read the complete file into the buffer
assign internal weight pointers at documented offsets
mark model valid
~~~

Do not assign a pointer before its containing region has passed bounds checks.

## Error behavior

Use a stable status convention rather than printing from low-level code:

~~~text
success
invalid argument
I/O failure
malformed header
unsupported configuration
truncated or oversized file
allocation failure
~~~

The loader should return failure and leave the output model handle null or destroyable. It should not call exit.

## Tests and gates

### Configuration tests

- valid TinyStories values derive head_dim=48 and kv_dim=288;
- zero dimensions fail;
- dim % n_heads != 0 fails;
- invalid KV-head relationships fail;
- unsupported untied classifier fails for the MVP.

### Checkpoint tests

- tiny valid fixture loads;
- expected final offset equals fixture size;
- one-byte truncation fails;
- extra trailing byte fails;
- invalid header field fails;
- tied classifier points to the embedding region;
- failed allocation/read paths clean up;
- loaded pointers lie within the owned buffer.

### Completion gate

This phase is complete only when:

- make test passes under strict warnings and sanitizers;
- the Python inspector and C loader report identical metadata and offsets;
- malformed fixtures fail without leaks or crashes;
- model destruction is safe after successful and partial loads;
- no activation or forward-pass code is required yet.

