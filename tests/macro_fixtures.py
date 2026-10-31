"""Small macro tokenizers and Fraise configs shared by hierarchical tests."""

from datasets import Dataset
from fixtures import tiny_tokenizer

from fraise.config.hierarchical import FraiseConfig
from fraise.tokenizer.macro_training import train_macro_tokenizer
from fraise.tokenizer.training import train_tokenizer

MACRO_TEXTS = [
    "a b c d e a b c d e a b",
    "a b c a b c d e e d c b a",
    "c d e c d e a b a b",
] * 4


def tiny_macro_tokenizer(vocab_size=14, max_macro_length=3):
    return train_macro_tokenizer(
        tiny_tokenizer(),
        {"fixture": Dataset.from_dict({"text": MACRO_TEXTS})},
        vocab_size,
        max_macro_length=max_macro_length,
    )


BYTE_TEXTS = [
    "The straw hat and the strawberry field.\nStraw bales stood by the barn.",
    "A strawberry is red. The field had straw and strawberries.\nThe end.",
    "Bits per byte compare models that use different tokenizers.",
] * 8


def byte_level_tokenizers(macro_vocab_size=320, max_macro_length=4):
    """A real byte-level BPE base and macro merges over it."""
    source = {"fixture": Dataset.from_dict({"text": BYTE_TEXTS})}
    base = train_tokenizer(source, 280)
    macro = train_macro_tokenizer(
        base, source, macro_vocab_size, max_macro_length=max_macro_length
    )
    return base, macro


def tiny_fraise_config(tokenizer, **changes):
    return FraiseConfig(
        **(
            {
                "vocab_size": tokenizer.get_vocab_size(),
                "base_vocab_size": tokenizer.base_vocab_size,
                "max_macro_length": tokenizer.max_macro_length,
                "residual_stream_dim": 8,
                "head_dim": 4,
                "qk_dim": 4,
                "num_heads": 2,
                "swiglu_up_dim": 16,
                "num_layers": 1,
                "max_seq_len": 8,
                "macro_residual_dim": 2,
                "local_dim": 8,
                "local_num_layers": 1,
                "local_swiglu_up_dim": 8,
            }
            | changes
        )
    )
