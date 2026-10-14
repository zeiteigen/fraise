"""Key/value cache for incremental decoding."""

import torch


class LayerCache:
    """Keys and values seen so far by one attention layer."""

    def __init__(self):
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

    def __len__(self) -> int:
        return 0 if self.keys is None else self.keys.size(2)

    def extend(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append ``[batch, heads, new, dim]`` entries and return the full history."""
        if self.keys is not None:
            keys = torch.cat((self.keys, keys), dim=2)
            values = torch.cat((self.values, values), dim=2)
        self.keys, self.values = keys, values
        return keys, values


class KVCache:
    def __init__(self, num_layers: int):
        self.layers = [LayerCache() for _ in range(num_layers)]

    def __len__(self) -> int:
        return len(self.layers[0])
