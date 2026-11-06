"""Fine-tuning the router model (M3).

- core.py: pure functions (validation split, loss mask, choosing epochs and threshold),
  tested in CI without torch.
- bundle.py: the bundle a training run produces, and its checker.
- train.py: the training script itself (torch, transformers, peft), run on Colab.
"""
