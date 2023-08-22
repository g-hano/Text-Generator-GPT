# GPT Shakespearean Text Generator

This repository contains a PyTorch implementation of a Shakespearean text generator using a simplified version of the GPT (Generative Pre-trained Transformer) architecture. This implementation is based on Andrej Karpathy's tutorial on building a text generator from scratch.

## Model Details

- **Model Size**: 10.799695 million parameters.
- **Training Progress**:
  - **Step 0**: Train loss 4.3666, Validation loss 4.3557
  - **Step 500**: Train loss 1.9409, Validation loss 2.0472
  - **Step 999**: Train loss 1.2576, Validation loss 1.6341

## Overview

The text generator is built using a series of neural network modules, including self-attention layers, feedforward layers, and a language model head. The key components of the model are as follows:

### Head and Multi-Head Attention

The `Head` class represents one head of self-attention, and the `MultiHeadAttention` class combines multiple heads of self-attention in parallel. These modules are essential for capturing dependencies between words in the text.

### FeedForward

The `FeedForward` class defines a simple linear layer followed by a non-linearity. This component helps in learning complex patterns in the data.

### Block

The `Block` class represents a Transformer block, which consists of multi-head self-attention followed by feedforward layers. The Transformer architecture allows the model to capture long-range dependencies in the text.

### Bigram Language Model

The `BigramLanguageModel` class combines all the above components to create a language model. It uses token embeddings and position embeddings to represent the input text and then applies a series of Transformer blocks. The model can be used for both text generation and text completion tasks.

## Usage

To use this Shakespearean text generator, follow these steps:

1. Import the necessary modules:

   ```python
   from generator import BigramLanguageModel
   ```

2. Instantiate the `BigramLanguageModel` class:

   ```python
   model = BigramLanguageModel()
   ```

3. Train the model on your Shakespearean text dataset (not included in this repository).

4. Generate text using the `generate` method:

   ```python
   context = torch.zeros((1, 1), dtype=torch.long, device=device)
   generated_text = decode(m.generate(context, max_new_tokens=500)[0].tolist())
   print(generated_text)
   ```

   This will generate a Shakespearean-style text continuation based on the provided input.

## Training

Training the model on your own dataset requires a substantial amount of text data and compute resources. You can fine-tune the model using techniques such as backpropagation and gradient descent to adapt it to your specific text generation task.

## Dependencies

- PyTorch
- TorchText (for data preprocessing, not included in this repository)
- Other common Python libraries (NumPy, etc.)

## Acknowledgments

- Andrej Karpathy's [tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4434s) served as the foundation for this project.
- The model architecture is based on the Transformer architecture introduced in the paper "Attention is All You Need" by Vaswani et al.
