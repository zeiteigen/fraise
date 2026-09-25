
from datasets_mgr.babylm import use_babylm
from datasets_mgr.finewebedu import use_fineweb_edu

from datasets_mgr import unified_dataset_loader

from components.model import ModelConfig

DATASET_LOADER = unified_dataset_loader([use_babylm(), use_fineweb_edu()])

TRAINING_HYPERPARAMETERS = {
    "num_epochs": 1,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_steps": 1000,
    "sequence_length": 512,
    # The checked-in local corpus currently contains roughly 0.27B packed
    # target tokens. Increase this only after adding and fingerprinting more
    # data; the trainer refuses to silently stop below this cap.
    "max_train_tokens": 250_000_000,
    "max_validation_tokens": 5_000_000,
    "packing_stride": 512,
    "seed": 1337,
    "checkpoint_interval_steps": 1000,
    "tokenization_batch_size": 1000,
    "use_bf16": True,
    "shuffle": True,
    "gradient_clip_norm": 1.0,
}

MODEL_CONFIG = ModelConfig(
    # ``load_tokenizer`` requires a concrete requested vocabulary size. The
    # returned tokenizer's actual size is still used by the model below.
    vocab_size=8192,
    residual_stream_dim=512,
    head_dim=64,
    qk_dim=64,
    num_heads=8,
    swiglu_up_dim=2048,
    num_layers=6,
    max_seq_len=TRAINING_HYPERPARAMETERS["sequence_length"],
)
