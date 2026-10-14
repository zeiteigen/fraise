import unittest
from copy import deepcopy
from dataclasses import MISSING, asdict, fields

from fixtures import tiny_config, tiny_settings

from fraise.config.training import TrainingSettings
from fraise.training.optimizer import OPTIMIZER_IDENTITY
from fraise.training.run.manifest import check_resume_identity


class ResumeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.current = {
            "manifest_version": 2,
            "identity": {
                "optimizer": OPTIMIZER_IDENTITY,
                "model_config": asdict(tiny_config(tie_word_embeddings=True)),
                "training": asdict(tiny_settings()),
                "token_budget_scope": "per_epoch",
                "dataset": {},
                "tokenizer": {},
            },
        }

    def test_runs_from_earlier_optimizers_cannot_resume(self):
        for optimizer in (None, "adamw", "muon-adamw", OPTIMIZER_IDENTITY):
            with self.subTest(optimizer=optimizer):
                saved = deepcopy(self.current)
                if optimizer is not None:
                    saved["identity"]["optimizer"] = optimizer
                else:
                    del saved["identity"]["optimizer"]
                if optimizer == OPTIMIZER_IDENTITY:
                    check_resume_identity(saved, self.current)
                else:
                    with self.assertRaisesRegex(ValueError, "optimizer.*fresh run"):
                        check_resume_identity(saved, self.current)

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

    def test_old_muon_runs_default_to_compiled_nondeterministic_execution(self):
        saved = deepcopy(self.current)
        for field in (
            "use_compile",
            "compile_threads",
            "deterministic",
            "log_interval_steps",
        ):
            del saved["identity"]["training"][field]
        check_resume_identity(saved, self.current)
        for field, value in (("use_compile", False), ("deterministic", True)):
            changed = deepcopy(self.current)
            changed["identity"]["training"][field] = value
            with self.assertRaisesRegex(ValueError, "resume training"):
                check_resume_identity(saved, changed)

    def test_reporting_interval_and_compiler_workers_can_change_on_resume(self):
        changed = deepcopy(self.current)
        changed["identity"]["training"]["compile_threads"] = 2
        changed["identity"]["training"]["log_interval_steps"] = 10
        check_resume_identity(self.current, changed)

    def test_operational_settings_can_change_on_resume(self):
        changed = deepcopy(self.current)
        changed["identity"]["training"].update(
            checkpoint_interval_steps=7,
            checkpoint_keep_last=9,
            tokenization_batch_size=3,
            validation_cpu_threads=1,
            validation_device="npu",
            validation_timeout_seconds=12.5,
        )
        check_resume_identity(self.current, changed)

    def test_every_defaulted_setting_may_be_missing_from_an_older_manifest(self):
        defaulted = [
            field.name
            for field in fields(TrainingSettings)
            if field.default is not MISSING and field.name != "minibatch_size"
        ]
        for name in defaulted:
            with self.subTest(name=name):
                saved = deepcopy(self.current)
                saved["identity"]["training"].pop(name)
                current = deepcopy(self.current)
                current["identity"]["training"][name] = getattr(TrainingSettings, name)
                if name == "packing_stride":
                    current["identity"]["training"][name] = current["identity"][
                        "training"
                    ]["sequence_length"]
                check_resume_identity(saved, current)

    def test_json_round_trip_of_tuple_settings_still_matches(self):
        saved = deepcopy(self.current)
        saved["identity"]["training"]["betas"] = list(
            saved["identity"]["training"]["betas"]
        )
        check_resume_identity(saved, self.current)


if __name__ == "__main__":
    unittest.main()
