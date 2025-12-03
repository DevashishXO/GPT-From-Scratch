# GPT From Scratch: Complete Implementation Journey

## Project Overview

### Goal

To implement a GPT (Generative Pre-trained Transformer) language model from scratch, following Andrej Karpathy's tutorial as a foundation, and then extend it with custom experiments including hyperparameter tuning, advanced tokenization, and multiple attention mechanism implementations.

### Author

Devashish Nagpal & Archita Bhargava

### Timeline

September 2025 - December 2025

---

## Table of Contents

1. [Dataset Description](#dataset-description)
2. [Phase 1: Foundation Building](#phase-1-foundation-building)
3. [Phase 2: First GPT Implementation](#phase-2-first-gpt-implementation)
4. [Phase 3: Modified GPT with BPE Tokenization](#phase-3-modified-gpt-with-bpe-tokenization)
5. [Phase 4: Production-Ready Colab Script](#phase-4-production-ready-colab-script)
6. [Key Learnings &amp; Conclusions](#key-learnings--conclusions)

---

## Dataset Description

### Source

**Hindi Literature Dataset** - A curated collection of Hindi text sourced from classic literature, stored at `dataset/new_input.txt`.

### Statistics

| Metric | Value |

|--------|-------|

| Total Characters | 6,793,953 |

| Language | Hindi (Devanagari Script) |

| Content Type | Literary prose, dialogues, narratives |

### Challenges Faced

#### 1. **Unicode and Encoding Issues**

-**Problem**: Hindi text uses Devanagari script with complex Unicode characters, diacritics, and conjuncts.

-**Solution**: Ensured UTF-8 encoding throughout the pipeline with explicit `encoding="utf-8"` in all file operations.

#### 2. **Character-Level vs Subword Tokenization**

-**Problem**: Initial character-level tokenization resulted in:

- Very long sequences (inefficient)
- Poor semantic understanding (each character has no meaning)
- Large vocabulary for covering all Unicode points

-**Solution**: Implemented BPE (Byte-Pair Encoding) tokenization which:

- Creates meaningful subword units
- Reduces sequence length significantly
- Handles out-of-vocabulary words gracefully

#### 3. **Vocabulary Size Selection**

-**Problem**: Too small vocabulary = poor coverage; too large = sparse embeddings

-**Solution**: Experimented with vocab sizes from 5,000 to 10,000, settled on **8,000** as optimal for this corpus size.

#### 4. **Train/Validation Split**

-**Problem**: Random splitting could break sentence/p		aragraph boundaries

-**Solution**: Used simple 90/10 sequential split to maintain text coherence within splits.

---

## Phase 1: Foundation Building

### 1.1 Self-Attention Mathematical Implementation

Before implementing the full GPT, I implemented the self-attention mechanism mathematically to understand the core concepts.

#### The Self-Attention Formula

```

Attention(Q, K, V) = softmax(QK^T / √d_k) × V

```

#### Implementation

```python

# Mathematical Self-Attention from scratch

wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)  # QK^T / √d_k

wei = wei.masked_fill(tril == 0, float('-inf'))       # Causal masking

wei = F.softmax(wei, dim=-1)                          # Softmax

out = wei @ v                                          # Multiply by V

```

#### Key Parameters Introduced

| Parameter | Symbol | Description |

|-----------|--------|-------------|

| Query Dimension | `d_q` | Dimension of query vectors |

| Key Dimension | `d_k` | Dimension of key vectors (= d_q for dot-product attention) |

| Value Dimension | `d_v` | Dimension of value vectors |

| Model Dimension | `d_model` | Overall embedding dimension |

| Head Size | `head_size` | d_model / n_heads |

#### Why Scaling by √d_k?

- Without scaling, dot products grow large with dimension
- Large values push softmax into regions with tiny gradients
- Scaling keeps variance stable: `Var(q·k) = d_k` → after scaling: `Var = 1`

### 1.2 Positional Encoding Understanding

Since transformers process all positions in parallel (unlike RNNs), we need to inject position information:

```python

self.position_embedding_table = nn.Embedding(block_size, n_embd)

pos_emb = self.position_embedding_table(torch.arange(T, device=device))

x = tok_emb + pos_emb  # Add position information to token embeddings

```

---

## Phase 2: First GPT Implementation

### 2.1 Model Architecture

#### Hyperparameter Configuration

| Hyperparameter | Symbol | Value | Rationale |

|----------------|--------|-------|-----------|

| Batch Size | `B` | 64 | Balance between gradient stability and memory |

| Block Size (Context) | `T` | 256 | Maximum sequence length for attention |

| Model Dimension | `d_model` | 384 | Embedding and hidden dimension |

| Number of Heads | `n_heads` | 6 | Parallel attention patterns |

| Head Dimension | `d_k = d_q = d_v` | 64 | 384 / 6 = 64 per head |

| Number of Layers | `n_layer` | 6 | Depth of transformer stack |

| Dropout | `p` | 0.2 | Regularization rate |

| Vocabulary Size | `vocab_size` | 95 | Character-level (unique chars in dataset) |

#### Architecture Diagram

```

Input Tokens [B, T]

       ↓

Token Embedding [B, T, 384] + Position Embedding [T, 384]

       ↓

┌─────────────────────────────────────┐

│         Transformer Block ×6        │

│  ┌─────────────────────────────┐   │

│  │ LayerNorm                    │   │

│  │ Multi-Head Attention (6 heads)│  │

│  │ Residual Connection          │   │

│  └─────────────────────────────┘   │

│  ┌─────────────────────────────┐   │

│  │ LayerNorm                    │   │

│  │ Feed-Forward (384→1536→384)  │   │

│  │ Residual Connection          │   │

│  └─────────────────────────────┘   │

└─────────────────────────────────────┘

       ↓

Final LayerNorm

       ↓

Linear Head [B, T, vocab_size]

       ↓

Output Logits

```

#### Total Parameters: **10,788,959** (~10.8M)

### 2.2 Training Configuration

| Parameter | Value |

|-----------|-------|

| Optimizer | AdamW |

| Learning Rate | 3e-4 (with cosine decay) |

| Min Learning Rate | 3e-5 |

| Warmup Iterations | 2000 |

| Max Iterations | 10,000 |

| Weight Decay | 0.1 |

| Gradient Clipping | 1.0 |

| Mixed Precision (AMP) | Enabled |

### 2.3 Training Results

#### Loss Progression

| Iteration | Train Loss | Val Loss | Val Perplexity | Learning Rate |

|-----------|------------|----------|----------------|---------------|

| 0 | 4.5798 | 4.5765 | 97.19 | 0.00e+00 |

| 250 | 2.4869 | 2.5372 | 12.64 | 3.75e-05 |

| 500 | 2.2188 | 2.3050 | 10.02 | 7.50e-05 |

| 750 | 2.0335 | 2.1598 | 8.67 | 1.12e-04 |

| 1000 | 1.8988 | 2.0521 | 7.78 | 1.50e-04 |

| 2000 | 1.4817 | 1.7063 | 5.51 | 3.00e-04 |

| 3000 | 1.1910 | 1.5658 | 4.79 | 3.00e-04 |

| 4000 | 0.9939 | 1.5092 | 4.52 | 2.95e-04 |

| 5000 | 0.8369 | 1.5093 | 4.52 | 2.89e-04 |

| 7500 | 0.5418 | 1.6291 | 5.10 | 2.57e-04 |

| 10000 | 0.3433 | 1.8137 | 6.13 | 2.10e-04 |

#### Loss Curve Analysis

**Key Observations:**

1.**Rapid Initial Learning (Steps 0-2000)**

- Train loss drops from 4.58 to 1.48 (68% reduction)
- Val loss drops from 4.58 to 1.71 (63% reduction)
- Perplexity improves from 97 to 5.5 (17x improvement)

2.**Optimal Point (Steps 3000-5000)**

- Best validation loss: **1.5092** at step 4000
- Best validation perplexity: **4.52**
- Train-val gap is healthy (~0.5)

3.**Overfitting Phase (Steps 5000+)**

- Train loss continues dropping (0.84 → 0.34)
- Val loss starts increasing (1.51 → 1.81)
- Classic overfitting pattern: model memorizing training data

4.**Interpretation**

- The model has ~10.8M parameters but only ~6.8M characters
- Character-level tokenization means fewer unique patterns to learn
- Should have stopped training at step ~4000-5000

### 2.4 Generated Sample Output

```

भीष्म साहनी उर्फ अलबर्ट, जो पहले एक जमाने में संसार की किसी कोने की हो सकती । मगर सिर्फ कुछ नहीं कि वह चीनी हक

बारे में देखता था, सोचता, 'ये कैसा आदमी है? पत्नी की इस तरह टोह लेना... ये तो हीनता है। भला कोई अपनी पत्नी का

```

**Analysis:**

- ✅ Grammatically correct Hindi sentences
- ✅ Proper punctuation and dialogue structure
- ✅ Coherent narrative style
- ✅ Character names and context preserved
- ⚠️ Some incomplete sentences (expected for character-level model)

---

## Phase 3: Modified GPT with BPE Tokenization

### 3.1 Key Modifications from Phase 2

| Aspect | Phase 2 | Phase 3 | Rationale |

|--------|---------|---------|-----------|

| Tokenization | Character-level (95 tokens) | BPE (10,000 subwords) | Better semantic units |

| Vocabulary Size | 95 | 10,000 | Cover more meaningful units |

| Special Tokens | None | [PAD], [UNK], [BOS], [EOS] | Proper sequence handling |

| Training Data | Raw characters | BPE-encoded sequences | Shorter, meaningful sequences |

### 3.2 BPE Tokenizer Implementation

```python

from tokenizers import Tokenizer, models, trainers, pre_tokenizers

from tokenizers.processors import TemplateProcessing


# Initialize BPE tokenizer

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()


# Train on corpus

trainer = trainers.BpeTrainer(

    vocab_size=10000,

    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

)

tokenizer.train(["dataset/new_input.txt"], trainer)


# Add BOS/EOS processing

tokenizer.post_processor = TemplateProcessing(

    single="[BOS] $A [EOS]",

    pair="[BOS] $A [EOS] [BOS] $B [EOS]",

    special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)]

)

```

### 3.3 Updated Hyperparameters

| Hyperparameter | Phase 2 | Phase 3 | Change |

|----------------|---------|---------|--------|

| `vocab_size` | 95 | 10,000 | +105× |

| `n_embd` (d_model) | 384 | 384 | Same |

| `n_head` | 6 | 6 | Same |

| `d_k = d_q = d_v` | 64 | 64 | Same |

| `n_layer` | 6 | 6 | Same |

| `dropout` | 0.2 | 0.2 | Same |

| `block_size` | 256 | 256 | Same |

### 3.4 Training Results (Partial - Run Interrupted)

| Step | Train Loss | Val Loss | Val PPL | LR |

|------|------------|----------|---------|-----|

| 0 | 9.3572 | 9.3532 | 11558.23 | 0.00e+00 |

| 250 | 6.6945 | 6.8109 | 906.89 | 3.75e-05 |

| 500 | 6.2313 | 6.4412 | 627.48 | 7.50e-05 |

| 750 | 5.9418 | 6.1878 | 486.83 | 1.12e-04 |

**Note**: This run was interrupted, but shows the characteristic higher initial loss when using BPE (more tokens = harder initial task) followed by rapid improvement.

### 3.5 Key Insight: Why Higher Initial Loss with BPE?

With character-level tokenization (vocab=95):

- Initial random guess: `log(95) ≈ 4.55` (matches observed ~4.58)

With BPE tokenization (vocab=10,000):

- Initial random guess: `log(10000) ≈ 9.21` (matches observed ~9.36)

This is expected and normal - the model must learn to predict from a much larger vocabulary.

---

## Phase 4: Production-Ready Colab Script

### 4.1 Major Enhancements

| Feature | Description |

|---------|-------------|

| **Google Drive Integration** | All checkpoints saved to Drive for persistence |

| **Early Stopping** | Stops training when validation loss stops improving |

| **Best Checkpoint Tracking** | Automatically saves and tracks best model |

| **Improved Regularization** | Higher dropout (0.4), label smoothing (0.1) |

| **Proper AdamW Groups** | No weight decay on LayerNorm and biases |

| **GELU Activation** | Replaced ReLU with GELU in feed-forward |

| **AMP Stability Fixes** | Fixed attention mask for fp16 stability |

### 4.2 Final Hyperparameter Configuration

| Hyperparameter | Symbol | Value |

|----------------|--------|-------|

| Vocabulary Size | `vocab_size` | 8,000 |

| Model Dimension | `d_model` | 384 |

| Number of Heads | `n_heads` | 6 |

| Head Dimension | `d_k = d_q = d_v` | 64 |

| Number of Layers | `n_layer` | 6 |

| Dropout | `p` | 0.4 |

| Embedding Dropout | `p_emb` | 0.1 |

| Block Size | `T` | 256 |

| Batch Size | `B` | 64 |

| Gradient Accumulation | - | 1 |

| Learning Rate | `lr` | 3e-4 |

| Min Learning Rate | `lr_min` | 3e-5 |

| Warmup Steps | - | 2,000 |

| Max Iterations | - | 25,000 |

| Weight Decay | `λ` | 0.1 |

| Label Smoothing | `ε` | 0.1 |

| Early Stop Patience | - | 6 evals |

### 4.3 Training Results

| Step | Train Loss | Train PPL | Val Loss | Val PPL | LR | Status |

|------|------------|-----------|----------|---------|-----|--------|

| 0 | 9.0770 | 8751.40 | 9.0733 | 8719.22 | 0.00e+00 | Start |

| 1000 | 5.8889 | 361.01 | 6.1432 | 465.52 | 1.50e-04 | ⬇️ Improving |

| 2000 | 5.2928 | 198.89 | 5.7911 | 327.37 | 3.00e-04 | ⬇️ Improving |

| 3000 | 4.9508 | 141.29 | 5.6268 | 277.76 | 2.99e-04 | ⬇️ Improving |

| 4000 | 4.7515 | 115.75 | 5.5647 | 261.05 | 2.95e-04 | ⬇️ Improving |

| 5000 | 4.6062 | 100.10 | 5.5248 | 250.85 | 2.89e-04 | ⬇️ Improving |

| 6000 | 4.4824 | 88.44 | 5.5111 | 247.42 | 2.80e-04 | ⬇️ Improving |

| 7000 | 4.3840 | 80.16 | 5.4884 | 241.88 | 2.70e-04 | ⬇️ Improving |

| 8000 | 4.2876 | 72.79 | 5.4587 | **234.80** | 2.57e-04 | **Best** |

| 9000 | 4.2220 | 68.17 | 5.4846 | 240.95 | 2.43e-04 | ⬆️ Rising |

| 10000 | 4.1706 | 64.75 | 5.4762 | 238.93 | 2.27e-04 | ➡️ Plateau |

| 12000 | 4.0592 | 57.93 | 5.4737 | 238.34 | 1.92e-04 | ➡️ Plateau |

| 14000 | 3.9842 | 53.74 | 5.5063 | 246.24 | 1.56e-04 | **Early Stop** |

**Total Training Time**: ~47 minutes

### 4.4 Loss Curve Interpretation

#### Phase Analysis

1.**Warm-up Phase (Steps 0-2000)**

- Learning rate ramps from 0 to 3e-4
- Both losses drop rapidly
- Model learning basic token patterns

2.**Peak Learning Phase (Steps 2000-8000)**

- Steepest improvement in validation loss
- Train loss: 5.29 → 4.29 (19% reduction)
- Val loss: 5.79 → 5.46 (6% reduction)
- Healthy generalization

3.**Plateau/Overfitting Phase (Steps 8000-14000)**

- Train loss continues dropping (4.29 → 3.98)
- Val loss stops improving, slight increase
- Early stopping triggered at step 14000

#### Why Early Stopping Worked

-**Best val loss**: 5.4587 (ppl 234.80) at step 8000

-**Final val loss**: 5.5063 (ppl 246.24) at step 14000

-**Saved 11,000 iterations** of unnecessary training

-**Prevented overfitting** before it became severe

### 4.5 Generated Sample

```

अब एक दिन तक उसने अपनी दुर्बलता का अनुभव किया था और वह केवल अपने 

प्रेम की रक्षा के सामने सिर झुका , उसका देव तुल पित तलवार और तलवार ें , 

उसे बे मौका ही न था ।

```

**Translation**: "Now for one day he experienced his weakness and he only bowed his head before the protection of his love, his god... sword and swords, he had no opportunity."

**Quality Assessment**:

- ✅ Grammatically correct Hindi
- ✅ Coherent narrative structure
- ✅ Emotional/literary tone matching training data
- ⚠️ Some semantic drift towards the end
- ⚠️ BPE artifacts (spaces before punctuation)

---

## Key Learnings & Conclusions

### 1. Tokenization Matters Significantly

| Tokenization | Vocab Size | Initial PPL | Best Val PPL | Semantic Quality |

|--------------|------------|-------------|--------------|------------------|

| Character-level | 95 | 97.19 | 4.52 | Low (no word meaning) |

| BPE | 8,000-10,000 | 8,700+ | 234-298 | High (subword meaning) |

**Insight**: Lower perplexity doesn't always mean better quality. Character-level achieves lower PPL but produces less meaningful text.

### 2. Regularization is Critical for Small Datasets

| Regularization | Effect |

|----------------|--------|

| Dropout 0.2 → 0.4 | Reduced overfitting significantly |

| Label Smoothing 0.1 | Prevented overconfident predictions |

| Early Stopping | Saved compute, preserved best model |

| Weight Decay (proper) | Stable training, better generalization |

### 3. Hyperparameter Sensitivity

Most impactful hyperparameters (in order):

1.**Learning Rate**: Too high = unstable, too low = slow

2.**Dropout**: Critical for generalization

3.**Number of Layers**: More = more capacity but more overfitting risk

4.**Vocabulary Size**: Affects both speed and quality

### 4. Training Dynamics Understanding

```

Early Training:  Loss drops fast (learning patterns)

Mid Training:    Loss drops slowly (refining)

Late Training:   Train drops, Val rises (overfitting)

                 → Stop here!

```

### 5. Practical Insights

-**AMP (Mixed Precision)**: 2-3x speedup with minimal quality loss

-**Gradient Accumulation**: Enables larger effective batch sizes

-**Checkpointing to Drive**: Essential for Colab (runtime can disconnect)

-**Cosine LR Schedule**: Smooth decay prevents sudden drops

---

## Summary Statistics

### Final Model Comparison Across Phases

| Metric | Phase 2 (Char) | Phase 4 (BPE) |

|--------|----------------|---------------|

| Parameters | 10.8M | 10.3M |

| Vocabulary | 95 | 8,000 |

| Best Val Loss | 1.51 | 5.46 |

| Best Val PPL | 4.52 | 235 |

| Training Time | ~33 min | ~47 min |

| Early Stopping | No | Yes (step 14k) |

| Output Quality | Moderate | Good |

### Key Hyperparameters Reference

```

d_model     = 384      # Embedding dimension

n_heads     = 6        # Attention heads  

d_k = d_q   = 64       # Key/Query dimension (384/6)

d_v         = 64       # Value dimension

n_layer     = 6        # Transformer blocks

d_ff        = 1536     # Feed-forward hidden (4 × d_model)

block_size  = 256      # Maximum context length

dropout     = 0.4      # Dropout probability

vocab_size  = 8000     # BPE vocabulary

```

---

## Appendix: File Structure

```

GPT-From-Scratch/

├── dataset/

│   └── new_input.txt          # Hindi literature corpus

├── tokenizer/

│   └── bpe_tokenizer.json     # Trained BPE tokenizer

├── checkpoints/

│   ├── best.pt                # Best validation checkpoint

│   ├── gpt_final.pt           # Final model

│   └── checkpoint_*.pt        # Periodic checkpoints

├── Implementing_GPT_from_Scratch!.ipynb  # Phase 1-3 experiments

├── GPT_from_Scratch_Attempt_2_0.ipynb    # Phase 4 + Attention comparison

├── colab_script.py            # Production training script

└── documentation.md           # This file

```

---

*Document generated for academic presentation - December 2024*
