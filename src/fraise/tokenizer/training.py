"""Train byte-level BPE from bounded batches of explicit text sources."""

from collections.abc import Iterator

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from fraise.config.values import integer

from .loading import SPECIAL_TOKENS, special_token_id

MIN_FREQUENCY = 2


def iter_text_batches(
    dataset: Dataset, *, batch_size=1000, max_bytes: int | None = None
) -> Iterator[list[str]]:
    if max_bytes is not None:
        integer("max_bytes", max_bytes, 0)
        if max_bytes == 0:
            return
    seen = 0
    for batch in dataset.select_columns("text").iter(batch_size=batch_size):
        texts = []
        for text in batch["text"]:
            if not isinstance(text, str):
                if text is None:
                    continue
                raise TypeError("tokenizer training text must be a string")
            if not text:
                continue
            texts.append(text)
            seen += len(text.encode("utf-8"))
            if max_bytes is not None and seen >= max_bytes:
                break
        if texts:
            yield texts
        if max_bytes is not None and seen >= max_bytes:
            return


def train_tokenizer(
    sources: dict[str, Dataset],
    vocab_size: int,
    *,
    max_bytes_per_source: int | None = None,
) -> Tokenizer:
    integer("vocab_size", vocab_size, 256 + len(SPECIAL_TOKENS))
    if not sources or not any(len(source) for source in sources.values()):
        raise ValueError("tokenizer training requires non-empty text sources")
    if max_bytes_per_source is not None:
        integer("max_bytes_per_source", max_bytes_per_source)
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=MIN_FREQUENCY,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=ByteLevel.alphabet(),
    )

    def batches():
        for source in sources.values():
            yield from iter_text_batches(source, max_bytes=max_bytes_per_source)

    tokenizer.train_from_iterator(batches(), trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        special_tokens=[
            (token, special_token_id(tokenizer, token)) for token in SPECIAL_TOKENS[1:]
        ],
    )
    return tokenizer
