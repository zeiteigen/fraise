from numbers import Integral

from torch import nn


class LMHead(nn.Module):
    """Project transformer hidden states into vocabulary logits.

    The final dimension of the input must be ``io_dim``. Any leading
    dimensions are preserved, so an input shaped ``(B, T, io_dim)`` produces
    logits shaped ``(B, T, vocab_size)``.
    """

    def __init__(self, io_dim, vocab_size, bias=True):
        super().__init__()

        for name, value in (("io_dim", io_dim), ("vocab_size", vocab_size)):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        if not isinstance(bias, bool):
            raise ValueError(f"bias must be a bool; got {bias!r}")

        self.io_dim = int(io_dim)
        self.vocab_size = int(vocab_size)
        self.projection = nn.Linear(self.io_dim, self.vocab_size, bias=bias)

    def forward(self, x):
        """Return unnormalized vocabulary logits for each input position."""
        return self.projection(x)
