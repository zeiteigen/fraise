"""Cross-checks between a tokenizer, model dimensions, and training settings."""

from fraise.architectures import ModelConfigProtocol, get_architecture

from .training import TrainingSettings


def check_compatibility(
    tokenizer, config: ModelConfigProtocol, settings: TrainingSettings
):
    if settings.sequence_length > config.max_seq_len:
        raise ValueError("sequence_length cannot exceed model max_seq_len")
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("tokenizer/model vocabulary mismatch")
    check_tokenizer = get_architecture(config.architecture).check_tokenizer
    if check_tokenizer is not None:
        check_tokenizer(config, tokenizer)
