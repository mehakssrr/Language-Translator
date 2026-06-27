# Seq2Seq Machine Translation (PyTorch)

A minimal, from-scratch implementation of a sequence-to-sequence (Encoder–Decoder) neural machine translation model using GRUs in PyTorch, with teacher forcing during training.

## Overview

This project implements the classic Encoder–Decoder architecture for sequence-to-sequence learning:

- **Encoder** — a GRU-based RNN that reads the source sequence and produces a final hidden state summarizing it.
- **Decoder** — a GRU-based RNN that takes the encoder's hidden state and generates the target sequence one token at a time.
- **Seq2Seq wrapper** — combines the encoder and decoder, handling the token-by-token decoding loop and teacher forcing.

The included `main.py` ships with a tiny built-in toy vocabulary/dataset (English → French phrases) so the script runs end-to-end out of the box. Swap in your own vocabulary and dataset to train on real data.

## Features

- Encoder/Decoder GRU architecture
- Teacher forcing with configurable ratio
- Padding-aware batching via a custom `collate_fn` (`pad_sequence`)
- Gradient clipping to stabilize training
- Automatic GPU/CPU device selection
- Loss masking on `<pad>` tokens via `ignore_index`
- Model checkpoint saving (`translation_model.pt`)

## Requirements

- Python 3.8+
- PyTorch
- torchvision

## Installation

```bash
pip install torch torchvision
```

## Usage

Run the training script directly:

```bash
python main.py
```

By default this trains on a tiny built-in toy dataset for 10 epochs and saves the trained weights to `translation_model.pt` in the working directory.

## Project Structure

```
.
├── main.py   # Encoder, Decoder, Seq2Seq model, training loop, and demo data
└── README.md
```

## How It Works

1. **Vocabulary & Data** — `input_vocab` and `output_vocab` map tokens to integer indices, including special tokens (`<pad>`, `<sos>`, `<eos>`). The demo `data` list holds a couple of toy (source, target) tensor pairs.
2. **Dataset & DataLoader** — `TranslationDataset` wraps the raw data; `collate_fn_pad_sequences` pads variable-length sequences within each batch so they can be stacked into a tensor.
3. **Encoder** — embeds the source tokens and runs them through a GRU, producing a final hidden state.
4. **Decoder** — starting from the `<SOS>` token and the encoder's hidden state, generates one token at a time, using either the ground-truth previous token (teacher forcing) or its own previous prediction.
5. **Training loop** — computes `NLLLoss` (ignoring padding), backpropagates, clips gradients, and steps the optimizer for each batch and epoch.
6. **Checkpoint** — after training, the model's weights are saved to `translation_model.pt`.

## Customization

To train on your own data:

- Replace `input_vocab` / `output_vocab` with vocabularies built from your dataset.
- Replace the `data` list with your own list of `(input_tensor, target_tensor)` pairs (token index sequences, including `<sos>`/`<eos>`).
- Tune hyperparameters in `main()`: `hidden_size`, `learning_rate`, `batch_size`, `num_epochs`, `teacher_forcing_ratio`.

## Notes

- This is an educational/reference implementation (no attention mechanism), intended as a clear starting point for understanding seq2seq translation models rather than a production-ready system.
- For longer sequences or larger vocabularies, consider adding an attention mechanism (e.g., Bahdanau or Luong attention) to improve translation quality.

## License

MIT — feel free to use, modify, and share.
