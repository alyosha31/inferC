# InferC MVP model contract

## Purpose

InferC's first supported model is deliberately narrow. The runtime supports
one checkpoint and tokenizer pairing before it attempts general model-format
support.

## Pinned artifacts

| Artifact | Local path | SHA-256 |
| --- | --- | --- |
| TinyStories 15M FP32 checkpoint | `models/stories15M.bin` | `cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a` |
| llama2.c tokenizer | `models/tokenizer.bin` | `50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361` |

The checkpoint comes from the `karpathy/tinyllamas` repository and uses the
binary layout consumed by `karpathy/llama2.c`. It is an FP32, decoder-only,
Llama-style model.

## Header

The checkpoint starts with seven little-endian signed 32-bit integers
(`28` bytes total):

| Field | Value | Meaning |
| --- | ---: | --- |
| `dim` | 288 | model/residual-stream width |
| `hidden_dim` | 768 | SwiGLU intermediate width |
| `n_layers` | 6 | transformer layers |
| `n_heads` | 6 | query heads |
| `n_kv_heads` | 6 | key/value heads |
| `vocab_size` | 32000 | positive means embedding/classifier weights are tied |
| `seq_len` | 256 | maximum context length |

Derived values:

```text
head_dim = dim / n_heads = 48
kv_dim   = n_kv_heads * head_dim = 288
```

Because `n_kv_heads == n_heads`, this target uses ordinary multi-head
attention, not grouped-query attention. Because `vocab_size > 0`, its final
classifier is the token embedding matrix reused as `W_vocab`.

## Serialized FP32 weight layout

All regions after the header are contiguous FP32 values. The offsets below
are byte offsets from the beginning of `stories15M.bin`.

| Region | Logical shape | Offset | Bytes |
| --- | --- | ---: | ---: |
| token embedding table | `[vocab_size, dim]` | 28 | 36,864,000 |
| attention RMSNorm | `[n_layers, dim]` | 36,864,028 | 6,912 |
| WQ | `[n_layers, dim, dim]` | 36,870,940 | 1,990,656 |
| WK | `[n_layers, kv_dim, dim]` | 38,861,596 | 1,990,656 |
| WV | `[n_layers, kv_dim, dim]` | 40,852,252 | 1,990,656 |
| WO | `[n_layers, dim, dim]` | 42,842,908 | 1,990,656 |
| FFN RMSNorm | `[n_layers, dim]` | 44,833,564 | 6,912 |
| W1 / gate | `[n_layers, hidden_dim, dim]` | 44,840,476 | 5,308,416 |
| W2 / down | `[n_layers, dim, hidden_dim]` | 50,148,892 | 5,308,416 |
| W3 / up | `[n_layers, hidden_dim, dim]` | 55,457,308 | 5,308,416 |
| final RMSNorm | `[dim]` | 60,765,724 | 1,152 |
| legacy RoPE real region | `[seq_len, head_dim / 2]` | 60,766,876 | 24,576 |
| legacy RoPE imaginary region | `[seq_len, head_dim / 2]` | 60,791,452 | 24,576 |

Expected checkpoint size: `60,816,028` bytes. The downloaded checkpoint
matches exactly.

The two legacy RoPE regions exist in this format for compatibility with older
exporters. The reference runtime skips them and computes RoPE angles during
the forward pass; InferC should do the same in the first implementation.

## Tokenizer contract

`tokenizer.bin` begins with the maximum token length as one little-endian
signed 32-bit integer. It then stores exactly `vocab_size` entries, each as:

```text
float32 score
int32 byte_length
byte_length raw token bytes
```

The tokenizer file does not independently declare vocabulary size; InferC
must validate it against the checkpoint configuration while loading.

## Scope boundaries

The MVP supports only:

- little-endian FP32 llama2.c-compatible checkpoints;
- the artifact pair and shapes documented above;
- tied token embedding/classifier weights;
- a 256-token context limit;
- standard multi-head attention.

Any support for alternate precision, untied classifiers, grouped-query
attention, or a different tokenizer format is a later deliberate extension,
not an implicit loader feature.

## Evidence command

Use this command after replacing either artifact:

```bash
shasum -a 256 models/stories15M.bin models/tokenizer.bin
```
