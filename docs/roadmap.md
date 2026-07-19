# InferC MVP phase plan

## Objective

Build a learning-first, scalar FP32 C17 runtime that runs one fixed model contract end to end:

```text
prompt text
-> matching tokenizer
-> token IDs
-> TinyStories 15M Llama-style forward passes
-> logits
-> greedy or temperature-based next-token selection
-> decoded generated text
```

The model contract is defined in [model-contract.md](model-contract.md). This roadmap prioritizes explainability and numerical correctness over speed or format generality.

## MVP scope

### Included

- scalar FP32 CPU execution in C17;
- one little-endian `llama2.c`-compatible TinyStories 15M checkpoint;
- matching `tokenizer.bin` BPE tokenizer;
- RMSNorm, softmax, RoPE, SiLU, SwiGLU, causal attention, and residual paths;
- standard multi-head attention for this checkpoint;
- per-session KV cache and one-token-at-a-time prefill/decode;
- deterministic operator and forward-logit comparison against Python;
- greedy decoding and temperature sampling;
- a standalone C CLI;
- sanitizer, warning, ownership, and documentation gates;
- basic load-time and tokens-per-second measurements.

### Explicitly excluded from the MVP

- model training or gradient computation;
- Hugging Face/safetensors/GGUF/general checkpoint support;
- alternate model sizes, untied classifiers, or GQA support;
- quantization, SIMD, BLAS, threading, GPU, batching, or memory mapping;
- generic tensor abstractions;
- concurrent sessions or production serving;
- cgo, Go HTTP, SSE, and cancellation.

Those are later extensions. They must not change the first runtime's data layout or delay proof of correct logits.

## Phase map

| Phase | Goal | Primary evidence | Unlocks |
| --- | --- | --- | --- |
| 0 | Freeze a real model contract | verified artifact hashes and weight offsets | all later shape/layout decisions |
| 1 | Complete scalar numerical primitives | Python-backed C unit tests | model loading and forward slices |
| 2 | Design immutable model loading | checkpoint inspection and offset validation | safe weight access |
| 3 | Design mutable session state | allocation/ownership tests and documented KV layout | allocation-free token forward path |
| 4 | Assemble one forward pass incrementally | intermediate activations and logit comparison | full inference correctness |
| 5 | Add tokenizer and generation | deterministic generation tests | usable CLI |
| 6 | Package and measure the MVP | repeatable build/run/benchmark instructions | post-MVP optimization or serving |

## Phase 0 — model contract

### Scope

- choose exactly one checkpoint/tokenizer pair;
- inspect header fields and calculate all tensor shapes;
- document serialized weight ordering and file-size formula;
- pin artifact hashes;
- define compatibility boundaries.

### Current result

This phase is complete. The chosen model has:

```text
dim         = 288
hidden_dim  = 768
layers      = 6
heads       = 6
kv_heads    = 6
head_dim    = 48
vocab       = 32000
context     = 256
```

### Completion gate

The calculated checkpoint size equals the actual file size, and every region has a documented logical shape and byte offset.

## Phase 1 — scalar numerical primitives

### Scope

Finish the small pure functions that the transformer will reuse:

1. matrix-vector multiplication — already implemented and tested;
2. RMSNorm — already implemented and tested;
3. stable softmax — already implemented and tested;
4. RoPE;
5. SiLU;
6. SwiGLU composition.

### Rules

Every operation must document:

- input and output logical shape;
- row-major flat-memory layout and indexing equation;
- read-only versus writable buffers;
- allocation behavior; these operators allocate nothing;
- time and extra-space complexity;
- numerical hazards and tolerance.

### Completion gate

Each operation has a hand-calculable case, a deterministic Python oracle, invalid-input coverage where applicable, and passes strict C17 plus AddressSanitizer/UBSan.

## Phase 2 — model configuration and checkpoint loading

### Scope

- define an immutable `ModelConfig` and validate its relationships;
- design an opaque model type;
- load the fixed binary header using checked file operations;
- calculate and validate all offsets using overflow-safe size arithmetic;
- allocate one model-owned checkpoint buffer;
- assign typed read-only pointers to weight regions;
- provide one destroy path that releases model-owned resources.

### Out of scope

- memory mapping;
- arbitrary checkpoint versions;
- silently accepting mismatched shapes;
- exposing weight pointers publicly.

### Completion gate

A checkpoint inspector and the C loader agree on every field and offset; truncated/malformed files fail safely; model loading and destruction pass sanitizers.

## Phase 3 — runtime session and memory layout

### Scope

Define a session object that owns all mutable inference state:

```text
x, normalized/residual buffers, q, attention output,
FFN intermediates, attention scores, logits, KV cache, position
```

The KV cache's logical shape is:

```text
[layer][position][kv_head][head_dim]
```

For the initial checkpoint, `kv_head == query_head`, but the logical naming remains explicit so the reason for each dimension stays clear.

### Required design decisions

- exact flattened KV indexing equation;
- activation-buffer reuse plan;
- maximum allocation size checks;
- session reset behavior;
- model lifetime versus session lifetime;
- partial-allocation cleanup behavior.

### Completion gate

A session can be created, reset, and destroyed without leaks; no forward-path operation will need `malloc` or `free`.

## Phase 4 — forward pass, built as vertical slices

### Scope and order

Implement only one independently verifiable slice at a time:

1. embedding lookup;
2. pre-attention RMSNorm and Q/K/V projections;
3. RoPE and KV-cache writes;
4. one-head causal score calculation, softmax, and value retrieval;
5. all heads plus output projection and attention residual;
6. FFN RMSNorm, SwiGLU, and FFN residual;
7. one complete transformer layer;
8. all six layers, final RMSNorm, and tied vocabulary projection.

### Validation strategy

Python must emit deterministic reference values for selected stages:

```text
embedding, normalized state, Q/K/V, rotated Q/K,
attention probabilities, attention output, FFN output,
layer output, final logits
```

Debug the first divergence; do not use generated text as the first signal of correctness.

### Completion gate

For fixed token IDs and positions, C and Python agree on documented intermediate values and final logits within an agreed FP32 tolerance. Record maximum absolute error, cosine similarity, and top-k agreement.

## Phase 5 — tokenizer, sampler, and C CLI

### Scope

- load the tokenizer format paired with the checkpoint;
- implement prompt encoding and safe token-piece decoding;
- run prompt prefill sequentially to populate the KV cache;
- implement greedy decoding first;
- add temperature sampling with a fixed seed for tests;
- expose a small CLI that accepts model path, tokenizer path, prompt, step count, temperature, and seed.

### Out of scope

- general tokenizer formats;
- top-p until greedy and temperature are proven correct;
- HTTP or streaming.

### Completion gate

Known prompts encode to expected IDs, fixed-seed generation is repeatable, the CLI produces text without sanitizer failures, and the KV cache persists across decode steps.

## Phase 6 — documentation and baseline measurement

### Scope

- document the execution path, ownership model, and KV-cache layout;
- provide build, test, and run commands;
- measure model load time, time to first token, and decode tokens/second;
- record machine/compiler/build flags with the measurements;
- write the first project retrospective: one implementation bug, one numerical-validation result, and one measured bottleneck.

### Completion gate

Another developer can clone the project, acquire the documented artifacts, run tests, inspect the model contract, and reproduce one generation command.

## Cross-phase engineering gates

Do not advance a core phase unless all applicable conditions hold:

1. no strict-warning failures (`-Wall -Wextra -Wpedantic -Werror`);
2. sanitizer tests pass;
3. numerical evidence exists where arithmetic is involved;
4. ownership and cleanup paths are documented;
5. dimensions and indexing can be stated precisely;
6. the developer can explain the relevant loop and allocation.

## Post-MVP sequence

After the C CLI meets Phase 6's gate, choose one extension deliberately:

1. public opaque C API;
2. Go/cgo HTTP and SSE serving;
3. profile-guided scalar-loop optimization;
4. GQA support using a new compatible checkpoint;
5. quantization research.

No post-MVP extension begins without a baseline benchmark and a correctness comparison plan.

