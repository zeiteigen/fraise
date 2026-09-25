from config import DATASET_LOADER, TRAINING_HYPERPARAMETERS, MODEL_CONFIG
from tokenizer import load_tokenizer
from train import pretrain_transformer
from components import FraiseModel


def main():


    # Load the tokenizer
    tokenizer = load_tokenizer(vocab_size=MODEL_CONFIG["vocab_size"], dataset=DATASET_LOADER)

    # Pretrain the transformer model
    pretrain_transformer(
        transformer_model_class=FraiseModel,
        tokenizer=tokenizer,
        dataset=DATASET_LOADER,
        hyperparameters=TRAINING_HYPERPARAMETERS,
        model_config=MODEL_CONFIG,
        save_path="/data/models"
    )