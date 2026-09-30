"""Small real tokenizers, model configs, and named sources for behavioral tests."""

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from fraise.config.model import ModelConfig
from fraise.config.training import TrainingSettings
from fraise.data.types import TrainingData


def tiny_tokenizer():
    vocab = {
        "<|pad|>": 0,
        "<|bos|>": 1,
        "<|eos|>": 2,
        "a": 3,
        "b": 4,
        "c": 5,
        "d": 6,
        "e": 7,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<|pad|>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="<|bos|> $A <|eos|>", special_tokens=[("<|bos|>", 1), ("<|eos|>", 2)]
    )
    return tokenizer


def tiny_config(**changes):
    return ModelConfig(
        **(
            {
                "vocab_size": 8,
                "residual_stream_dim": 8,
                "head_dim": 4,
                "qk_dim": 4,
                "num_heads": 2,
                "swiglu_up_dim": 16,
                "num_layers": 1,
                "max_seq_len": 4,
            }
            | changes
        )
    )


def tiny_settings(**changes):
    return TrainingSettings.from_dict(
        {
            "batch_size": 1,
            "learning_rate": 1e-3,
            "dataset_token_counts": {"fixture": 4},
            "max_train_tokens": 4,
            "num_epochs": 1,
            "sequence_length": 2,
            "checkpoint_interval_steps": 1,
            "shuffle_per_epoch": False,
        }
        | changes
    )


def tiny_data(validation=False):
    source = Dataset.from_dict({"text": ["a b c d e"]})
    return TrainingData({"fixture": source}, source if validation else None)
