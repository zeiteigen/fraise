from typing import TypedDict

from torch import nn
from components import device
import torch.nn.functional as F

from train.save_mgr import SaveManager


import torch


from datasets import DatasetDict
from tokenizers import Tokenizer


class TrainingHyperparameters(TypedDict):
    """Hyperparameters for training a model."""

    batch_size: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    warmup_steps: int




def pretrain_transformer(transformer_model_class: nn.Module, tokenizer: Tokenizer, dataset: DatasetDict, hyperparameters: TrainingHyperparameters, model_config: dict, save_path: str, model_name: str = "fraise_model"):
    """
    Train a model using the provided datasets and hyperparameters.

    Args:
        model_class: The class of the model to be trained.
        dataset: A DatasetDict containing the training and validation datasets.
        hyperparameters: A dictionary containing training hyperparameters.
        model_config: A dictionary containing the model configuration.
        save_path: The path where the trained model will be saved.
    """

    model = transformer_model_class(model_config).to(device)

    save_manager = SaveManager(save_path, model_name)

    save_manager.log(f"Starting training for {hyperparameters['num_epochs']} epochs with batch size {hyperparameters['batch_size']} and learning rate {hyperparameters['learning_rate']}.")


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters["learning_rate"],
        weight_decay=hyperparameters["weight_decay"],
    )


    intermediary_messages = []

    for epoch in range(hyperparameters["num_epochs"]):

        model.train()
        for batch in dataset["train"].iter(batch_size=hyperparameters["batch_size"]):

            # Tokenize the input text
            tokenized_inputs = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt")

            # Move batch to the appropriate device
            tokenized_inputs = tokenized_inputs.to(device)

            input = tokenized_inputs[:, :-1]
            target = tokenized_inputs[:, 1:]

            optimizer.zero_grad(set_to_none=True)

            logits = model(input)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
            )

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            intermediary_messages.append(
                f"Epoch {epoch + 1}/{hyperparameters['num_epochs']}, Batch Loss: {loss.item()}, Grad Norm: {grad_norm.item()}"
            )
            
        
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation step
        model.eval()
        with torch.no_grad():
            for batch in dataset["validation"]:
                tokens = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt")
                tokens = tokens.to(device)

                target = tokens[:, 1:]
                logits = model(tokens[:, :-1])

                val_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target.reshape(-1),
                )
                intermediary_messages.append(f"Validation Loss: {val_loss.item()}")

        intermediary_messages.append(f"Epoch {epoch + 1}/{hyperparameters['num_epochs']}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

        save_manager.log("\n".join(intermediary_messages))

        # Save the model checkpoint
        save_manager.save_checkpoint(model, optimizer)

    save_manager.finish()

     




    