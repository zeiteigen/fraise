import unittest
from copy import deepcopy
from dataclasses import asdict

from fixtures import tiny_config, tiny_settings

from fraise.training.manifest import check_resume_identity


class ResumeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.current = {
            "manifest_version": 2,
            "identity": {
                "model_config": asdict(tiny_config(tie_word_embeddings=True)),
                "training": asdict(tiny_settings()),
                "token_budget_scope": "per_epoch",
                "dataset": {},
                "tokenizer": {},
            },
        }

    def test_older_training_config_can_resume_with_default_initialization(self):
        saved = deepcopy(self.current)
        del saved["identity"]["training"]["tied_weight_init_std"]
        check_resume_identity(saved, self.current)

    def test_changed_initialization_is_rejected_for_old_and_new_configs(self):
        changed = deepcopy(self.current)
        changed["identity"]["training"]["tied_weight_init_std"] = 0.1
        for omit in (False, True):
            with self.subTest(omit=omit):
                saved = deepcopy(self.current)
                if omit:
                    del saved["identity"]["training"]["tied_weight_init_std"]
                with self.assertRaisesRegex(ValueError, "resume training"):
                    check_resume_identity(saved, changed)

    def test_changed_model_dimensions_are_still_rejected(self):
        changed = deepcopy(self.current)
        changed["identity"]["model_config"]["residual_stream_dim"] *= 2
        with self.assertRaisesRegex(ValueError, "resume model_config"):
            check_resume_identity(self.current, changed)

    def test_legacy_shuffle_resumes_with_equivalent_unmixed_settings(self):
        for shuffled in (False, True):
            with self.subTest(shuffled=shuffled):
                current = deepcopy(self.current)
                current["identity"]["training"]["shuffle_per_epoch"] = shuffled
                saved = deepcopy(current)
                training = saved["identity"]["training"]
                training["shuffle"] = training.pop("shuffle_per_epoch")
                del training["shuffle_data"]
                check_resume_identity(saved, current)
                current["identity"]["training"]["shuffle_data"] = True
                with self.assertRaisesRegex(ValueError, "resume training"):
                    check_resume_identity(saved, current)

    def test_old_single_epoch_budget_is_compatible(self):
        saved = deepcopy(self.current)
        del saved["identity"]["token_budget_scope"]
        check_resume_identity(saved, self.current)

    def test_old_multi_epoch_budget_cannot_restore_a_shorter_schedule(self):
        current = deepcopy(self.current)
        current["identity"]["training"]["num_epochs"] = 3
        saved = deepcopy(current)
        del saved["identity"]["token_budget_scope"]
        with self.assertRaisesRegex(ValueError, "start a fresh run"):
            check_resume_identity(saved, current)


if __name__ == "__main__":
    unittest.main()
