# Japanese → English Neural Machine Translation (Transformer)

A Transformer encoder-decoder for Japanese-to-English translation, implemented in PyTorch and trained on the KFTT dataset. The whole pipeline (data preparation, tokenization, model, training, decoding and BLEU evaluation) lives in a single notebook.

## Results

Evaluated on the KFTT test split.

| Decoding | BLEU |
|---|---|
| Greedy | 20.54 |
| Beam search | 21.49 |

Beam search gives about +1 BLEU over greedy decoding. Outputs are slightly short (brevity penalty 0.90), so tuning the length penalty is a clear next step.

## Dataset

[KFTT (Kyoto Free Translation Task)](http://www.phontron.com/kftt/), a Japanese-English parallel corpus built from Wikipedia articles about Kyoto, loaded through Hugging Face `datasets`. It uses the standard train / validation / test splits. [Add: number of training pairs and any filtering applied.]

## Model

- Encoder-decoder built on `nn.Transformer` with separate source and target embeddings
- d_model 256, 4 attention heads, 3 encoder + 3 decoder layers, feed-forward size 1024 (~40M parameters)
- Pre-LayerNorm (`norm_first=True`) for more stable training
- Sinusoidal positional encoding
- Tied target embedding and output projection weights
- Padding and causal masks, with teacher forcing during training

## Training

- Mixed precision (fp16)
- Dropout 0.2
- Averaged checkpoints for the final model
- Single T4 GPU, about 9 epochs
- [Add: optimizer, learning-rate schedule, batch size, label smoothing if used]

## Decoding

Greedy decoding and beam search, both evaluated with BLEU.

## Repository contents

- `nmt-ja-to-en.ipynb`: data preparation, model, training and evaluation, with outputs saved in the notebook

## How to run

1. Open the notebook in Google Colab or Jupyter with a GPU runtime.
2. The download dataset through Hugging Face `datasets`.
3. Run the cells in order.


## References

- Vaswani et al. *Attention Is All You Need*, 2017
- Neubig. *The Kyoto Free Translation Task*, 2011
