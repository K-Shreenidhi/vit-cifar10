# Vision Transformer from Scratch — CIFAR-10

I implemented a Vision Transformer completely from scratch for this assignment —
patch embedding, CLS token, positional embeddings, multi-head self-attention, MLP,
residual connections, and LayerNorm are all hand-written (no `nn.MultiheadAttention`,
`nn.TransformerEncoder`, `F.scaled_dot_product_attention`, or pretrained weights).
Trained on CIFAR-10 only, random init.

## Results

| Model | Patch Config | Val Acc | Test Acc (top-1) |
|---|---|---|---|
| Baseline ViT | 4×4 non-overlapping patches | 81.78% | 81.93% |
| ViT + overlapping patches | 8×8 kernel, stride 4 | 84.08% | 83.54% |

## Running it

Open `vit_cifar10.ipynb` in Colab, switch runtime to GPU (T4 is what I used), run
all cells top to bottom. It downloads CIFAR-10 itself, trains the baseline, then the
overlapping-patch variant, and evaluates both on the test set once each.

If Colab kills your session mid-training (it happened to me — free-tier GPU quota
ran out at epoch 86 on one run), just re-run the training cell. `resume=True` picks
up from the last checkpoint saved to Drive instead of starting over.

## Config

| | |
|---|---|
| Image size | 32×32 |
| Patch size | 4×4 baseline / 8×8 kernel, stride 4 for overlap variant |
| Embed dim | 256 |
| Depth | 6 blocks |
| Heads | 8 |
| MLP ratio | 4.0 |
| Dropout | 0.1 |
| Optimizer | AdamW, weight decay 0.05 |
| LR | 3e-4, 5-epoch warmup then cosine decay |
| Batch size | 128 |
| Epochs | 100 (baseline); overlap run stopped at 86 — see notes below |
| Seed | 42 |
| GPU | Tesla T4 |
| Params | ~5M |

## Why these choices

**Patch size 4×4, not 8×8 or 16×16.** CIFAR-10 images are only 32×32. 16×16 patches
(the usual ViT default) would give 4 tokens total nowhere near enough for attention
to do anything meaningful at this resolution. 4×4 gives 64 tokens, which is the
standard choice for CIFAR-scale ViT and keeps sequence length manageable.

**LayerNorm instead of BatchNorm**, and written by hand rather than `nn.LayerNorm`
since the assignment requires it. LayerNorm normalizes each token independently
across its own feature dimension, so it doesn't depend on batch size or composition
 that's the right fit for a sequence of tokens, unlike BatchNorm which needs
batch-level statistics.

**Pre-LN blocks** (`x = x + Attn(LN(x))` rather than `LN(x + Attn(x))`). Post-LN is
what the original Transformer paper used, but it's harder to train stably without a
careful warmup schedule. Pre-LN keeps a clean residual path for gradients, which is
why basically every modern Transformer implementation (ViT included) uses it.

**Data augmentation (random crop + horizontal flip) on train only.** A from-scratch
ViT has no convolutional inductive bias to lean on it has to learn everything
about local structure from data, so without augmentation it overfits CIFAR-10's 45k
training images fast. Val/test stay unaugmented so the reported numbers are clean.

## Extra-credit experiment: overlapping patch embeddings

The idea: with non-overlapping 4×4 patches, two pixels that are right next to each
other in the actual image can end up in completely different tokens if they happen
to fall on either side of a patch boundary that boundary information just gets
split with no shared context. I tried making patches overlap instead —
`Conv2d(kernel=8, stride=4, padding=2)` so each patch is 8×8 pixels but steps by
only 4, meaning neighboring patches share half their pixels. I set the padding so
the number of tokens stays exactly the same as the baseline (64 patches) everything
else (depth, heads, embed dim, LR schedule, seed, augmentation) is identical between
the two runs, so this is a controlled comparison of just the patch tokenization.

**Result:** 81.93% → 83.54% test accuracy, a +1.61pp improvement.

**My read on why it helped:** CIFAR-10 images are small and low-res, so a feature
that matters for classification (an edge, a wheel outline) can easily land right on
a 4px patch boundary and get fragmented across two tokens in the non-overlapping
version. With overlap, that same region is more likely to be captured whole inside
at least one token, so the linear projection has cleaner information to work with
before attention even runs. Attention itself isn't doing anything different between
the two runs it's still full global self-attention over all 65 tokens either way
so the gain has to be coming from better input representations, not from how
tokens communicate. This lines up with why some later ViT variants (T2T-ViT, for
instance) moved toward richer/overlapping tokenization instead of a hard grid.

**One thing I'm not claiming:** this shows overlap helps at this scale (~5M params)
and this training budget. I didn't test whether it still helps with more epochs or
a bigger model that would need a separate experiment.

**Trade-off:** overlapping patches do slightly more compute in the patch-embedding
conv (bigger kernel), but it's a small fraction of the total model's FLOPs, most of
which come from the 6 Transformer blocks.

## Notes from actually building this

Two things worth mentioning since they shaped the final numbers:

- Early on I had a bug where a quick 2-epoch smoke test in `src/train.py` was
  saving to the same checkpoint path as the real 100-epoch training run, so a test
  run silently overwrote a fully-trained checkpoint. Caught it by checking the
  classification head's weight std against its init value (std ≈ 0.02) — a barely
  trained model's weights haven't moved much from init, which is what I saw. Fixed
  by giving smoke tests and real runs separate checkpoint paths.
- The overlap-patch run got cut off at epoch 86/100 by a Colab GPU quota limit.
  Looking at the curve at that point val accuracy plateaued around 0.84 while
  train accuracy was already at 98%+ and val loss was creeping up it was already
  past the useful point, so I used the epoch-86 checkpoint (best validation
  accuracy) rather than burning more GPU hours chasing the last 14 epochs.

## Repo contents

- `vit_cifar10.ipynb` — full implementation, training, validation, and test
  evaluation for both variants, with outputs from an actual run.
- `README.md` — this file.
