import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from components.model import FraiseModel  # noqa: E402
from datasets_mgr import unified_dataset_loader  # noqa: E402
from train.training import pretrain_transformer  # noqa: E402


class TrainingRunTests(unittest.TestCase):
    def test_small_run_uses_source_map_and_retains_latest_checkpoint(self):
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
            single="<|bos|> $A <|eos|>",
            special_tokens=[("<|bos|>", 1), ("<|eos|>", 2)],
        )
        source = Dataset.from_dict({"text": ["a b c d e"]})
        source.info.dataset_name = "fixture"
        dataset = unified_dataset_loader([source], source_column=False)

        settings = {
            "batch_size": 1,
            "learning_rate": 1e-3,
            "dataset_token_counts": {"fixture": 4},
            "num_epochs": 1,
            "sequence_length": 2,
            "max_train_tokens": 4,
            "checkpoint_interval_steps": 1,
            "checkpoint_keep_last": 1,
            "shuffle": False,
        }
        model_config = {
            "vocab_size": len(vocab),
            "residual_stream_dim": 8,
            "head_dim": 4,
            "qk_dim": 4,
            "num_heads": 2,
            "swiglu_up_dim": 16,
            "num_layers": 1,
            "max_seq_len": 2,
        }

        with TemporaryDirectory() as directory:
            with patch.object(FraiseModel, "compile", lambda self: None):
                pretrain_transformer(
                    tokenizer=tokenizer,
                    dataset=dataset,
                    hyperparameters=settings,
                    model_config=model_config,
                    save_path=directory,
                    model_name="tiny",
                )
            run_dir = Path(directory) / "tiny"
            self.assertEqual(
                [path.name for path in run_dir.glob("checkpoint_step_*.pt")],
                ["checkpoint_step_00000002.pt"],
            )
            self.assertTrue((run_dir / "tiny.pt").is_file())


if __name__ == "__main__":
    unittest.main()
