# inferC
# InferC

**A minimal Llama-style transformer inference runtime written from scratch in C, with Python-based numerical validation and a streaming Go serving layer.**

InferC is an educational systems project focused on understanding how transformer inference actually works below high-level machine-learning frameworks.

The primary goal is not to build the fastest runtime or support every model architecture. The goal is to implement the critical inference path manually and understand:

* how model weights are represented in memory,
* how transformer operations are expressed as loops over flat arrays,
* how binary checkpoints are loaded,
* how attention uses a KV cache,
* how autoregressive token generation works,
* how memory ownership is managed in C,
* how a native runtime can be exposed through a Go service,
* and how numerical correctness can be verified against Python.

The core runtime is written in **C17**. Python acts as a correctness oracle and development tool. Go is used only after the C runtime can generate text independently.

---

# Project Status

InferC is currently in its initial implementation phase.

The first milestone is a scalar FP32 C runtime capable of generating text from a small Llama-style checkpoint.

The initial development target is a small TinyStories model so that the complete forward pass remains understandable, testable, and practical to run locally.

The project should be developed incrementally. A working C command-line runtime is more important than the Go server, advanced optimizations, or production infrastructure.

---

# Core Philosophy

InferC is a learning-first project.

The project intentionally avoids abstractions that would hide important details. The implementation should make memory layout, tensor dimensions, indexing, ownership, and computation visible.

The following principles guide development:

1. **Correctness before optimization**
2. **Explicit memory ownership**
3. **Flat arrays before generic tensor abstractions**
4. **Scalar loops before SIMD or threading**
5. **Small independently testable modules**
6. **Python reference outputs for numerical validation**
7. **One working vertical slice before expanding scope**
8. **The developer must understand every core line of the runtime**

Generated code should not replace understanding. Tools may assist with repetitive work, tests, build scripts, documentation, and review, but the primary runtime logic should be written and understood by the project owner.

---

# Goals

The first complete version of InferC should:

* load a small Llama-style FP32 checkpoint,
* load the corresponding tokenizer,
* encode a text prompt into token IDs,
* execute the transformer forward pass one token at a time,
* maintain a key-value cache across generated tokens,
* produce vocabulary logits,
* select the next token using greedy or probabilistic sampling,
* decode generated token IDs into text,
* expose generation through a C command-line interface,
* validate operator and logit correctness against Python,
* report basic inference performance metrics,
* expose generation through a Go HTTP API,
* stream generated text using Server-Sent Events.

---

# Non-Goals for the Initial Version

The following are intentionally out of scope for the MVP:

* CUDA or GPU execution,
* INT8 or INT4 quantization,
* SIMD intrinsics,
* a generic tensor library,
* arbitrary-rank tensors,
* broadcasting,
* automatic differentiation,
* model training,
* arbitrary Hugging Face model support,
* continuous batching,
* paged attention,
* concurrent inference sessions,
* distributed inference,
* Kubernetes deployment,
* production-grade authentication,
* production-grade observability,
* speculative decoding.

These may be explored later, but they must not delay the first correct end-to-end runtime.

---

# Technology Stack

## C17

C implements the inference runtime.

It is responsible for:

* checkpoint loading,
* model configuration,
* weight representation,
* runtime activation buffers,
* mathematical operators,
* transformer execution,
* KV caching,
* tokenization,
* sampling,
* generation state,
* and the public native API.

C is used because it exposes memory allocation, pointer arithmetic, data layout, file formats, and resource ownership directly.

## Python

Python is used as a development and validation tool.

It is responsible for:

* generating reference operator outputs,
* inspecting checkpoints,
* comparing C and Python results,
* calculating numerical error,
* generating test fixtures,
* and analysing benchmark results.

Python is not part of production inference.

## Go

Go is used only after the C runtime works from the command line.

It is responsible for:

* HTTP request handling,
* request validation,
* streaming responses,
* cancellation propagation,
* serialization,
* basic metrics,
* and calling the C runtime through a narrow cgo boundary.

The Go layer should not know the internal layout of model weights or runtime buffers.

---

# High-Level Architecture

```text
                         InferC

┌────────────────────────────────────────────────────────┐
│                                                        │
│  Client                                                │
│    │                                                   │
│    │ POST /v1/completions                              │
│    ▼                                                   │
│  Go HTTP Server                                        │
│    ├── request validation                              │
│    ├── request cancellation                            │
│    ├── SSE token streaming                             │
│    └── metrics                                         │
│              │                                         │
│              │ cgo                                     │
│              ▼                                         │
│  Public C API                                          │
│    ├── model loading                                    │
│    ├── session creation                                 │
│    ├── incremental token generation                     │
│    └── resource destruction                             │
│              │                                         │
│              ▼                                         │
│  C Inference Runtime                                   │
│    ├── tokenizer                                       │
│    ├── checkpoint loader                               │
│    ├── model weights                                   │
│    ├── runtime buffers                                 │
│    ├── transformer forward pass                        │
│    ├── KV cache                                        │
│    └── sampler                                         │
│                                                        │
│  Python Development Tools                              │
│    ├── operator references                             │
│    ├── checkpoint inspection                           │
│    ├── logit comparison                                │
│    └── benchmark analysis                              │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

# Repository Structure

```text
inferc/
├── runtime/
│   ├── include/
│   │   ├── inferc.h
│   │   ├── config.h
│   │   ├── checkpoint.h
│   │   ├── tokenizer.h
│   │   ├── ops.h
│   │   ├── transformer.h
│   │   ├── sampler.h
│   │   └── session.h
│   │
│   ├── src/
│   │   ├── config.c
│   │   ├── checkpoint.c
│   │   ├── tokenizer.c
│   │   ├── ops.c
│   │   ├── transformer.c
│   │   ├── sampler.c
│   │   ├── session.c
│   │   └── cli.c
│   │
│   ├── tests/
│   │   ├── test_matvec.c
│   │   ├── test_rmsnorm.c
│   │   ├── test_softmax.c
│   │   ├── test_rope.c
│   │   └── test_forward.c
│   │
│   └── Makefile
│
├── server/
│   ├── cmd/
│   │   └── inferd/
│   │       └── main.go
│   │
│   ├── internal/
│   │   ├── runtime/
│   │   │   ├── runtime.go
│   │   │   └── bindings.go
│   │   ├── api/
│   │   │   ├── handler.go
│   │   │   └── request.go
│   │   ├── stream/
│   │   │   └── sse.go
│   │   └── metrics/
│   │       └── metrics.go
│   │
│   └── go.mod
│
├── tools/
│   ├── reference_ops.py
│   ├── inspect_checkpoint.py
│   ├── compare_logits.py
│   ├── benchmark.py
│   └── requirements.txt
│
├── models/
│   └── .gitkeep
│
├── docs/
│   ├── architecture.md
│   ├── model-execution.md
│   ├── memory-layout.md
│   ├── benchmarks.md
│   └── roadmap.md
│
├── scripts/
│   ├── download_model.sh
│   ├── test.sh
│   └── run.sh
│
├── Makefile
├── README.md
├── .gitignore
└── LICENSE
```

Not every file must be created immediately. Modules should be introduced as they become necessary.

---

# C Module Responsibilities

## `config`

Defines the model architecture and validates relationships between model dimensions.

Expected configuration fields include:

* model dimension,
* feed-forward hidden dimension,
* number of transformer layers,
* number of attention heads,
* number of key-value heads,
* vocabulary size,
* and maximum sequence length.

This module should also derive or validate values such as:

```text
head_dim = model_dim / number_of_attention_heads
```

The configuration should be treated as immutable after model loading.

---

## `checkpoint`

Loads model configuration and weights from a binary checkpoint.

Responsibilities include:

* opening the checkpoint file,
* reading and validating the header,
* determining the expected file size,
* allocating or mapping weight memory,
* assigning typed pointers to individual weight regions,
* detecting truncated or incompatible files,
* and releasing all checkpoint resources.

The first implementation should use straightforward file operations such as `fopen`, `fread`, `fseek`, and `ftell`.

Memory mapping may be added only after the initial implementation works.

This module should make the checkpoint layout explicit and document every offset.

---

## `ops`

Contains the low-level numerical operations used by the transformer.

The initial implementation should provide scalar FP32 versions of:

* matrix-vector multiplication,
* RMS normalization,
* softmax,
* rotary positional embeddings,
* SiLU activation,
* and the operations required for SwiGLU.

The first important function is matrix-vector multiplication.

The implementation should:

* use flat row-major arrays,
* avoid dynamic allocation,
* receive all output buffers from the caller,
* document the shape of every input,
* and favour clarity over optimization.

Each operation should be independently validated against Python.

---

## `transformer`

Combines model weights, runtime buffers, and low-level operations into one transformer forward step.

For a token at a particular sequence position, the forward pass should perform:

```text
token embedding lookup
        ↓
for each transformer layer:
        RMSNorm
        query, key, and value projections
        rotary positional embedding
        key and value cache write
        causal attention over cached positions
        attention output projection
        residual connection
        RMSNorm
        SwiGLU feed-forward network
        residual connection
        ↓
final RMSNorm
        ↓
vocabulary projection
        ↓
logits
```

The first forward-pass milestone is:

> Given a token ID and position, the C runtime produces logits that numerically match a trusted reference within an acceptable tolerance.

Generated text should not be used as the first correctness test. Intermediate numerical values and final logits are more useful for debugging.

---

## `session`

Owns mutable inference state.

Responsibilities include:

* allocating activation buffers,
* allocating the key cache,
* allocating the value cache,
* allocating logits,
* tracking the current token position,
* resetting generation state,
* and freeing all session-owned memory.

Buffers should normally be allocated once and reused across tokens.

The forward pass should avoid calling `malloc` or `free`.

The session must clearly document:

* which memory it owns,
* which memory belongs to the model,
* which buffers are mutable,
* and when each allocation is released.

A model may eventually be shared across sessions because model weights are read-only. Mutable KV-cache state belongs to an individual session.

---

## `tokenizer`

Loads tokenizer data and converts between text and token IDs.

Responsibilities include:

* loading the vocabulary,
* encoding prompt text,
* decoding token IDs into token pieces,
* handling beginning-of-sequence tokens,
* handling unknown tokens,
* and correctly joining decoded pieces.

Tokenizer work must not distract from the transformer runtime. The initial tokenizer should closely follow the format expected by the selected checkpoint.

---

## `sampler`

Selects the next token from model logits.

The initial implementation should support:

* greedy decoding,
* temperature scaling,
* and top-p sampling.

Sampling should support deterministic tests by allowing a fixed random seed.

The sampler owns sampling configuration but should not own model logits.

---

## `inferc.h`

Defines the narrow public interface used by applications and the Go server.

Internal model and session structures should be opaque.

External callers should only know that model and session types exist. They should interact through functions for:

* model loading,
* model destruction,
* session creation,
* incremental generation,
* session reset,
* and session destruction.

The public API must not expose internal weight arrays, KV-cache pointers, or translated cgo-specific types.

---

# Encapsulation Model

InferC uses C-style encapsulation.

Public headers expose:

* opaque type declarations,
* public function declarations,
* public enums,
* and stable configuration types when necessary.

Private `.c` files contain:

* full struct definitions,
* private helper functions,
* private global constants,
* and implementation details.

Private helper functions should normally be marked `static`.

For example, callers may hold a pointer to an `InferModel`, but they should not know the fields inside the model.

This provides the C equivalent of a class with private fields and public methods.

---

# Memory Ownership

Memory ownership must be explicit throughout the project.

Every allocation should have a clearly identified owner.

Suggested ownership rules:

## Model-owned memory

The model owns:

* model configuration,
* checkpoint data,
* model weights,
* tokenizer vocabulary when shared across sessions,
* and other immutable model-level data.

It is released when the model is destroyed.

## Session-owned memory

A session owns:

* activation buffers,
* query, key, and value buffers,
* attention scores,
* hidden feed-forward buffers,
* logits,
* key cache,
* value cache,
* generation position,
* and session-specific sampling state.

It is released when the session is destroyed.

## Caller-owned memory

Public APIs may receive output buffers owned by the caller.

The function contract must state:

* required buffer size,
* whether the buffer may be modified,
* whether returned pointers remain valid,
* and whether the caller must free returned memory.

## General rules

* Every successful allocation must have exactly one corresponding release.
* Model weights should be treated as read-only after loading.
* No pointer should outlive the allocation it refers to.
* No function should return a pointer to a local stack variable.
* Allocation failure must be handled.
* Cleanup paths must release all previously acquired resources.
* Sanitizers must be used throughout development.

---

# Data Layout

InferC should avoid a generic tensor abstraction in the MVP.

Matrices and tensors should be represented as flat arrays with explicitly documented dimensions.

For a row-major matrix with `rows` rows and `cols` columns:

```text
element(row, col) = data[row * cols + col]
```

Every numerical function should document:

* logical shape,
* physical layout,
* indexing formula,
* mutable versus immutable buffers,
* and output dimensions.

This is essential for understanding checkpoint offsets, matrix multiplication, attention heads, and KV-cache layout.

---

# KV Cache

Autoregressive generation processes one new token at a time.

Keys and values computed for earlier positions do not need to be recomputed. They are stored in a KV cache.

Conceptually, the cache must distinguish:

* transformer layer,
* sequence position,
* key-value head,
* and head dimension.

The exact flattened indexing formula must be documented before implementation.

A session owns its KV cache because it changes as generation progresses.

The cache must not be shared across unrelated generation requests unless the sharing semantics are explicitly designed.

---

# Python Validation Strategy

Python acts as a correctness oracle.

The project should validate each operator independently before validating the complete transformer.

## Operator comparison

For each operator:

1. Generate deterministic inputs in Python.
2. Calculate the expected result using NumPy or PyTorch.
3. Run the C implementation on exactly the same input.
4. Compare element-wise output.
5. Report the maximum and mean absolute error.
6. Fail the test when the configured tolerance is exceeded.

Initial operators to validate:

1. matrix-vector multiplication,
2. RMS normalization,
3. softmax,
4. rotary positional embedding,
5. SiLU,
6. SwiGLU.

## Forward-pass comparison

The full forward pass should compare:

* selected intermediate activations,
* layer outputs,
* final logits,
* top-k token IDs,
* and cosine similarity between logit vectors.

Useful metrics include:

* maximum absolute error,
* mean absolute error,
* relative error,
* cosine similarity,
* top-1 token agreement,
* and top-k overlap.

Generated text alone is not a sufficient correctness test.

---

# Development Order

The runtime should be implemented in this order:

1. repository and build setup,
2. Python reference harness,
3. scalar matrix-vector multiplication,
4. RMS normalization,
5. numerically stable softmax,
6. rotary positional embeddings,
7. SiLU and SwiGLU operations,
8. model configuration representation,
9. checkpoint inspection,
10. checkpoint loader,
11. runtime buffer allocation,
12. embedding lookup,
13. query, key, and value projections,
14. KV-cache layout and writes,
15. causal attention,
16. attention output projection,
17. feed-forward network,
18. complete transformer layer,
19. final normalization and vocabulary projection,
20. full forward-pass comparison,
21. tokenizer loading and prompt encoding,
22. greedy token generation,
23. temperature and top-p sampling,
24. C command-line interface,
25. public C API,
26. Go cgo wrapper,
27. HTTP endpoint,
28. SSE streaming,
29. cancellation,
30. benchmarks and documentation.

Only one bounded milestone should be implemented at a time.

---

# Initial Milestones

## Milestone 1: Build a C program

Create a small executable with:

* a header file,
* a source implementation,
* a command-line entry point,
* compiler warnings,
* sanitizers,
* and a Makefile.

The purpose is to understand compilation, linking, headers, source files, include paths, and object ownership.

## Milestone 2: Implement and test `matvec`

The first numerical operation is matrix-vector multiplication.

Before implementation, the developer should be able to explain:

* matrix dimensions,
* input and output dimensions,
* row-major layout,
* the flattened indexing formula,
* time complexity,
* and which buffers are read-only or writable.

The implementation should not allocate memory.

## Milestone 3: Implement remaining primitive operations

Add RMSNorm, softmax, RoPE, SiLU, and SwiGLU with independent Python comparisons.

## Milestone 4: Load a checkpoint

Read model configuration and map every weight region.

Before writing this module, the binary file layout must be documented.

## Milestone 5: Produce correct logits

Execute a complete forward step and compare logits with a reference implementation.

## Milestone 6: Generate text from the C CLI

Encode a prompt, repeatedly run the forward pass, sample tokens, and decode them.

This is the most important MVP milestone.

## Milestone 7: Add the Go serving layer

Expose the working runtime through a narrow native API and stream generated tokens over HTTP.

---

# Build Configuration

Development builds should use strict warnings and sanitizers.

Recommended compiler configuration:

```text
C standard: C17

Warnings:
-Wall
-Wextra
-Wpedantic
-Werror

Debugging:
-g

Sanitizers:
-fsanitize=address,undefined
```

The include directory should be provided through the compiler include path rather than hard-coding paths such as `../include/header.h` throughout the source code.

For example, source files should use:

```c
#include "ops.h"
```

while the build system provides:

```text
-Iruntime/include
```

Release builds should be added only after correctness is established.

---

# Testing Requirements

Every core module requires tests.

Tests should cover:

* normal inputs,
* smallest valid inputs,
* invalid dimensions where applicable,
* null pointers where the API permits validation,
* numerical edge cases,
* extreme softmax inputs,
* sequence-boundary conditions,
* checkpoint truncation,
* allocation failure paths when practical,
* and deterministic sampling.

All C tests should run under address and undefined-behaviour sanitizers.

A milestone is not complete merely because it compiles.

A milestone is complete when:

1. the implementation compiles without warnings,
2. tests pass,
3. sanitizers report no errors,
4. numerical results match the reference,
5. ownership is documented,
6. and the developer can explain the implementation.

---

# CLI Target

The completed C runtime should support a command resembling:

```bash
./build/inferc \
  --model models/stories15M.bin \
  --tokenizer models/tokenizer.bin \
  --prompt "Once upon a time" \
  --max-tokens 100 \
  --temperature 0.8
```

The exact argument design may change.

The CLI should eventually report:

* generated text,
* prompt token count,
* generated token count,
* model load duration,
* time to first token,
* total generation duration,
* and tokens per second.

---

# Go Serving Target

The Go service should expose a small completion API.

Suggested endpoint:

```text
POST /v1/completions
```

Suggested request:

```json
{
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.8,
  "stream": true
}
```

Suggested streaming response:

```text
data: {"token":" there"}

data: {"token":" was"}

data: [DONE]
```

The initial server may serialize inference requests.

Correctness is more important than concurrent generation. A shared mutable session or KV cache must never be used concurrently without an explicit design.

The Go service should:

* load the model once at startup,
* keep C types hidden inside one internal package,
* translate C errors into Go errors,
* propagate request cancellation,
* free all session resources,
* and avoid exposing cgo types in public Go APIs.

---

# Performance Measurement

Performance should be measured only after correctness is established.

Initial metrics:

* model load time,
* prompt length,
* generated token count,
* time to first token,
* total inference time,
* tokens per second,
* checkpoint size,
* model weight memory,
* KV-cache memory,
* runtime activation memory,
* and peak resident memory where practical.

Benchmarks must document:

* CPU model,
* operating system,
* compiler,
* compiler flags,
* model,
* prompt,
* output length,
* and whether the build is debug or release.

Numbers should not be added to the README or résumé until measured.

---

# Learning Outcomes

By completing InferC, the developer should understand:

## C and systems programming

* compilation and linking,
* headers and source files,
* include paths,
* opaque structs,
* public and private module boundaries,
* pointers,
* arrays,
* pointer arithmetic,
* stack versus heap,
* dynamic allocation,
* ownership,
* cleanup,
* binary file I/O,
* error handling,
* and debugging with sanitizers.

## Numerical computing

* row-major memory layout,
* flattened multidimensional indexing,
* matrix-vector multiplication,
* floating-point accumulation,
* numerical tolerance,
* stable softmax,
* normalization,
* activation functions,
* and approximate equality.

## Transformer inference

* token embeddings,
* RMSNorm,
* query, key, and value projections,
* attention heads,
* grouped-query attention,
* causal masking,
* rotary positional embeddings,
* KV caching,
* residual connections,
* SwiGLU,
* vocabulary projection,
* logits,
* autoregressive decoding,
* temperature,
* and top-p sampling.

## Backend integration

* native-library boundaries,
* opaque handles,
* cgo,
* request cancellation,
* streaming HTTP responses,
* and basic inference metrics.

---

# Reference Implementations

Minimal educational transformer implementations may be studied as specifications and debugging references.

The most relevant reference is Karpathy's `llama2.c`, which demonstrates Llama-style inference in a compact C implementation.

Reference code should be used to:

* understand checkpoint formats,
* validate equations,
* compare execution order,
* and debug mismatches.

It should not simply be copied wholesale and presented as an original implementation.

Every copied or adapted fragment must be understood and attributed.

---

# Definition of Done for the MVP

The MVP is complete when:

* the C runtime loads a supported checkpoint,
* the tokenizer encodes a prompt,
* the transformer forward pass runs without sanitizer errors,
* the KV cache is used across token positions,
* C logits match a trusted reference within a documented tolerance,
* the C CLI generates coherent text,
* sampling supports at least greedy decoding and temperature,
* tests are repeatable,
* memory ownership is documented,
* the README contains build and run instructions,
* and measured benchmark results are included.

The Go layer is valuable but secondary.

A correct C runtime that generates text is a successful project even if the serving layer is unfinished.

---

# Future Roadmap

Possible extensions after the MVP:

* memory-mapped checkpoints,
* SIMD vectorization,
* multithreaded matrix operations,
* quantized weights,
* improved tokenizer support,
* multiple model architectures,
* concurrent sessions,
* request scheduling,
* batching,
* paged KV caching,
* prefix caching,
* speculative decoding,
* benchmark comparisons with established runtimes,
* and a Rust or C++ implementation for architectural comparison.

Each optimization should be accompanied by:

* a baseline measurement,
* an explanation of the bottleneck,
* a correctness test,
* and a measured result.

---

# Résumé Positioning

Only implemented and measured features should be claimed.

A completed MVP may support bullets such as:

> Built a CPU-only Llama-style transformer inference runtime in C, implementing RMSNorm, RoPE, grouped-query attention, SwiGLU feed-forward layers, autoregressive decoding, and KV caching.

> Validated numerical operator and logit correctness against Python reference implementations using absolute-error, cosine-similarity, and top-k agreement checks.

> Exposed native token generation through a streaming Go HTTP API using cgo, request cancellation, and model initialization at server startup.

Measured throughput, latency, and memory figures should be added only after benchmarks are run.

---

# Contribution and Development Rules

During the learning phase:

* core runtime code should be authored by the project owner,
* code-generation tools should not implement entire core modules,
* every function must have documented input and output shapes,
* every allocation must have an owner,
* every module should have an independent test,
* and no optimization should be introduced without a correct baseline.

A tool may assist with:

* Makefiles,
* shell scripts,
* test harness boilerplate,
* fixture generation,
* CI configuration,
* formatting,
* repetitive error-handling cleanup,
* documentation,
* and reviewing developer-written code.

The purpose of InferC is not merely to possess a working repository. The purpose is to acquire the ability to explain and recreate the runtime.

