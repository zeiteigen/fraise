# Fraise Proof of Concept

The initial Fraise experiment is designed to test hierarchical asymmetric tokenization at a scale small enough for rapid local iteration while large enough to produce meaningful language-model behavior.

The proof of concept is trained locally on an **RTX 5090 Laptop GPU with 24 GB of VRAM and 32 GB of system RAM**.

## Model
- **Dense transformer parameters:** ~50M
- **Hidden size (`d_model`):** 512
- **Base/output vocabulary:** 8,192
- **Macro/input vocabulary:** 65,536
- **Vocabulary asymmetry:** 8×
- **Macro residual dimension:** 32
- **Training precision:** BF16

The 8k output vocabulary is deliberately smaller than conventional modern LLM vocabularies. This makes the 64k input vocabulary an explicitly asymmetric component rather than allowing improvements to be explained simply by increasing vocabulary size everywhere.

Sparse lexical parameters are reported separately from the ~50M dense transformer parameters.

## Token Representation

Every macro token has a deterministic decomposition into tokens from the 8k base vocabulary.

Rather than assigning every macro token an independent 512-dimensional embedding, Fraise constructs its representation from:

1. a shared composition of its constituent base-token embeddings; and    
2. a learned **32-dimensional macro-specific residual**, projected into the 512-dimensional model space.

This gives common macro-tokens room to learn specialized representations while allowing rare macro-tokens to fall back primarily on information shared with their better-trained constituents.

## Autoregression

The transformer operates globally over 64k macro-tokens, but generation occurs through the 8k base vocabulary.

Each macro-token decomposition is stored in a trie. During generation, the local decoder emits base tokens while traversing this trie. Once a valid macro-token is completed, its macro representation is supplied as the next input to the global transformer.

**Multi-Token Prediction (MTP)** is introduced as an optimization to this local decoding stage, allowing multiple constituent base tokens to be proposed between expensive global transformer passes.

## Dynamic Decomposition

Macro-tokens can be expanded when their internal structure becomes relevant.

A token may therefore be represented at multiple resolutions:

```text
[strawberry]

[straw] [berry]

[s] [t] [r] [a] [w] [b] [e] [r] [r] [y]
```

This provides two related capabilities:

- **structural access:** spelling, morphology, numbers, identifiers, rare words, and other token-internal information can remain accessible despite aggressive macro-tokenization;
    
- **adaptive compute:** difficult spans can receive additional transformer positions and KV-cache entries while ordinary spans remain compressed.
    

The first implementation will introduce decomposition incrementally rather than attempting fully dynamic hierarchical KV behavior immediately.

## Training Data

The target training budget is approximately **40 base tokens per dense model parameter**.

For a 50M-parameter transformer:

50M×40≈2B50M \times 40 \approx 2B

so the model is trained on approximately **2 billion 8k-base-token equivalents** sampled from **FineWeb-Edu**.

The training budget is measured using the shared 8k base tokenizer rather than each model's native input tokenization. All architectures therefore see the **same underlying text**, even though macro-tokenized models may require substantially fewer global transformer positions to process it.

The selected training subset is fixed before experiments and reused across all baselines and ablations.

## Baselines

At minimum, the initial comparison includes:

|Model|Input Vocabulary|Output Vocabulary|Purpose|
|---|--:|--:|---|
|Base|8k|8k|Small-vocabulary conventional baseline|
|Symmetric|64k|64k|Larger-vocabulary conventional baseline|
|Fraise|64k|8k|Asymmetric architecture|

All models use approximately the same dense transformer parameter budget and are trained on the same underlying data.

Additional Fraise ablations isolate:

- independent vs. compositional macro representations;
    
- macro residuals on/off;
    
- macro residual dimensionality;
    
- trie-based autoregressive bridging;
    
- MTP;
    
- static vs. dynamic decomposition;
    
- decomposition budget.
    

## Evaluation

### Language Modeling

Token-level perplexity is not directly comparable across different tokenizations, so the primary tokenizer-independent language-model metric is **bits per byte (BPB)**.

Held-out loss is also tracked within each tokenizer configuration.

### Token-Internal Reasoning

Architecture-specific benchmarks include:

- **CharBench** for character-level reasoning;
    
- **CUTE** for orthographic and token-internal understanding;
    
- synthetic tests over spelling, morphology, numbers, identifiers, rare words, and unseen compositions.
    

Synthetic evaluation should distinguish between:

- frequent macro-tokens;
    
- rare macro-tokens;
    
- unseen macro combinations composed from known base tokens;
    
- entirely novel strings.
    

### Efficiency

Fraise is intended to change the quality–compute tradeoff, so evaluation also includes:

- global transformer positions per byte;
    
- macro-token compression ratio;
    
- decompositions per sequence;
    
- KV-cache entries per byte;
    
- quality versus decomposition budget;
    
- training throughput;
    
- inference throughput;
    
- peak VRAM usage.
    

A particularly important experiment is the **quality versus decomposition-compute curve**: how much additional performance can be recovered by selectively expanding increasingly large fractions of the input?

## Tech Stack

- **PyTorch** — model implementation and training    
- **PyTorch SDPA** — optimized causal attention
- **`torch.compile`** — compilation and kernel-overhead reduction
- **Hugging Face Tokenizers** — training and applying the 8k byte-level base tokenizer  
- **Custom hierarchical tokenizer** — constructing the 64k macro vocabulary over sequences of base-token IDs and maintaining decomposition trees/trie structure
- **Hugging Face Datasets / PyArrow** — FineWeb-Edu preprocessing
- **MLflow** — local experiment and ablation tracking
- **PyTorch Profiler** — framework and memory profiling
- **NVIDIA Nsight Systems** — GPU profiling
- **uv** — environment and dependency management

The model is implemented directly in PyTorch rather than through a high-level LLM training framework because the architecture requires custom composition, trie decoding, MTP, decomposition, and eventually variable-resolution KV-cache behavior.

## Implementation Order

The proof of concept is built incrementally:

1. 8k/8k baseline
    
2. 64k/64k baseline
    
3. static 64k-input / 8k-output Fraise
    
4. compositional macro embeddings
    
5. 32-dimensional learned macro residuals
    
6. trie-based autoregressive bridging
    
7. Multi-Token Prediction
    
8. dynamic decomposition
    
9. variable-resolution / hierarchical KV-cache experiments
    

Each stage should remain independently trainable and evaluable so architectural regressions can be isolated.

## Success Criteria

The PoC does not need to outperform substantially larger language models.

It succeeds if asymmetric hierarchical tokenization demonstrates a favorable tradeoff between:

- **language-model quality,**
    
- **global transformer computation,**
    
- **sequence/KV-cache length,**
    
- and **access to fine-grained token structure.**
    

In particular, Fraise should demonstrate whether macro-token compression can reduce the number of expensive global transformer steps without permanently sacrificing the information hidden inside those macro-tokens.