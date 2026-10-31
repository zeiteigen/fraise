"""Count an architecture's parameters without allocating its weight tensors."""

from fraise.architectures import ModelConfigProtocol, get_architecture


def estimate_parameter_count(config: ModelConfigProtocol) -> int:
    """Count model weights and biases from their configured dimensions."""
    return get_architecture(config.architecture).count_parameters(config)
