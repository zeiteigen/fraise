from pathlib import Path
import json
import time

from helpers.get_git_commit import get_git_commit

import torch

META_FILE_NAME = "model_meta.json"

class SaveManager:
    def __init__(self, save_dir: str, model_name: str):
        self.save_dir = Path(save_dir)
        self.model_name = model_name

        self.messages = []
        self.epochs_completed = 0

        self.save_dir.mkdir(parents=True, exist_ok=True)
        with open(self.save_dir / META_FILE_NAME, "w") as f:
            json.dump({"model_name": self.model_name, "start_time": time.time(), "git_commit": get_git_commit()}, f)

    def log(self, message: str, print_to_console: bool = True):
        self.messages.append(message)
        if print_to_console:
            print(message) 
        self._update_meta()

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        self.epochs_completed += 1

        checkpoint_path = (
            self.save_dir /
            f"checkpoint_epoch_{self.epochs_completed}.pt"
        )

        torch.save(
            {
                "epoch": self.epochs_completed,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )

        self.log(f"Checkpoint saved at {checkpoint_path}")
        self._update_meta()


    def _update_meta(self, custom_dict: dict | None = None):
        meta_path = self.save_dir / META_FILE_NAME
        with open(meta_path, "r") as f:
            meta_data = json.load(f)
        
        meta_data["epochs_completed"] = self.epochs_completed
        meta_data["last_updated"] = time.time()
        meta_data["all_messages"] = self.messages

        if custom_dict is not None:
            meta_data.update(custom_dict)
        
        with open(meta_path, "w") as f:
            json.dump(meta_data, f)


    def finish(self):
        if self.epochs_completed == 0:
            raise RuntimeError("No checkpoints have been saved.")

        latest_checkpoint = (
            self.save_dir /
            f"checkpoint_epoch_{self.epochs_completed}.pt"
        )

        checkpoint_data = torch.load(latest_checkpoint)
        model_state_dict = checkpoint_data["model_state_dict"]

        torch.save(
            model_state_dict,
            self.save_dir / f"{self.model_name}.pt"
        )

        self.log("Training finished. Meta data updated.")
        self._update_meta({"training_finished": True, "final_model_path": str(self.save_dir / f"{self.model_name}.pt"), "end_time": time.time()})