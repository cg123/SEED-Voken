import argparse, os, sys, datetime, glob, importlib
from torch.utils.data import random_split, DataLoader, Dataset

import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, LearningRateMonitor

# Import StatefulDataLoader for checkpoint/resume with streaming datasets
from torchdata.stateful_dataloader import StatefulDataLoader


class StreamingDatasetTracker(Callback):
    """
    Simple callback for logging dataloader state during training.
    StatefulDataLoader handles all the resume logic automatically.
    """

    def __init__(self):
        super().__init__()
        self.first_batch_logged = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Log first batch info for verification."""
        if not self.first_batch_logged:
            print(f"[Rank {trainer.global_rank}] First batch of epoch {trainer.current_epoch}:")
            print(f"  - batch_idx={batch_idx}")

            # Log batch hash for verification
            if isinstance(batch, dict) and "image" in batch:
                img = batch["image"][0] if len(batch["image"]) > 0 else None
                if img is not None:
                    batch_hash = f"{img.sum().item():.6f}"
                    print(f"  - batch_hash={batch_hash}")

            self.first_batch_logged = True

    def on_train_epoch_start(self, trainer, pl_module):
        """Reset logging flag for new epoch."""
        self.first_batch_logged = False
from lightning import seed_everything

from torch.utils.data.dataloader import default_collate as custom_collate

import torch
torch.set_float32_matmul_precision("high")

# Performance settings - set deterministic=True only if reproducibility is critical
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(L.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None, persistent_workers=True, prefetch_factor=2):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        self.persistent_workers = persistent_workers and self.num_workers > 0
        self.prefetch_factor = prefetch_factor if self.num_workers > 0 else None
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

        # Store dataloader state for StatefulDataLoader
        self._train_dataloader_state = None
        self._current_train_dataloader = None
        self._train_dataloader_state_schema = "per_rank_v1"

    @staticmethod
    def _dist_is_initialized():
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized()

    def _select_train_dataloader_state(self, state):
        if isinstance(state, dict) and state.get("_schema") == self._train_dataloader_state_schema:
            per_rank = state.get("by_rank", [])
            if self._dist_is_initialized():
                import torch.distributed as dist
                if "world_size" in state and state["world_size"] != dist.get_world_size():
                    print(
                        "[DataModule] Warning: DDP world_size changed since checkpoint; "
                        "dataloader resume may be inaccurate."
                    )
                rank = dist.get_rank()
                if rank < len(per_rank):
                    return per_rank[rank]
                print(f"[DataModule] Warning: No dataloader state for rank {rank}")
                return None
            if len(per_rank) > 0:
                return per_rank[0]
            return None
        return state

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        from torch.utils.data import IterableDataset
        self.datasets = dict()
        for k in self.dataset_configs:
            if "pretrain" not in self.dataset_configs[k]["target"]: ##laion should use webdataset
                self.datasets[k] = instantiate_from_config(self.dataset_configs[k])
            else:
                self.datasets[k] = instantiate_from_config(self.dataset_configs[k]).create_dataset()
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

        # Track which datasets are iterable
        self.is_iterable = {k: isinstance(ds, IterableDataset) for k, ds in self.datasets.items()}

    def state_dict(self):
        """Save datamodule state for checkpointing."""
        import random
        import numpy as np
        import torch

        state = {}

        # Save StatefulDataLoader state if available
        if self._current_train_dataloader is not None:
            try:
                dataloader_state = self._current_train_dataloader.state_dict()
                if self._dist_is_initialized():
                    import torch.distributed as dist
                    world_size = dist.get_world_size()
                    if world_size > 1:
                        gathered = [None for _ in range(world_size)]
                        dist.all_gather_object(gathered, dataloader_state)
                        dataloader_state = {
                            "_schema": self._train_dataloader_state_schema,
                            "by_rank": gathered,
                            "world_size": world_size,
                        }
                state['train_dataloader_state'] = dataloader_state
                print("[DataModule] Saved StatefulDataLoader state")
            except Exception as e:
                print(f"[DataModule] Warning: Could not save dataloader state: {e}")

        # Save RNG states for reproducibility
        state.update({
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
        })

        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state_all()

        return state

    def load_state_dict(self, state_dict):
        """Load datamodule state from checkpoint."""
        import random
        import numpy as np
        import torch

        # Store dataloader state to restore after dataloader creation
        raw_dataloader_state = state_dict.get('train_dataloader_state')
        self._train_dataloader_state = self._select_train_dataloader_state(raw_dataloader_state)
        if self._train_dataloader_state is not None:
            print("[DataModule] Loaded StatefulDataLoader state (will restore in train_dataloader)")

        # Restore RNG states immediately
        if state_dict.get("python_rng_state") is not None:
            random.setstate(state_dict["python_rng_state"])
            print("[DataModule] Restored Python RNG state")

        if state_dict.get("numpy_rng_state") is not None:
            np.random.set_state(state_dict["numpy_rng_state"])
            print("[DataModule] Restored NumPy RNG state")

        if state_dict.get("torch_rng_state") is not None:
            torch.set_rng_state(state_dict["torch_rng_state"])
            print("[DataModule] Restored PyTorch RNG state")

        if state_dict.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state_dict["cuda_rng_state"])
            print("[DataModule] Restored CUDA RNG state")

    def _train_dataloader(self):
        """
        Create train dataloader using StatefulDataLoader for streaming datasets.
        This enables automatic checkpoint/resume for iterable datasets.
        """
        # IterableDatasets (webdataset, streaming HF datasets) don't support shuffle
        is_iterable = self.is_iterable.get("train", False)

        if is_iterable:
            # Use StatefulDataLoader for streaming/iterable datasets
            dataloader = StatefulDataLoader(
                self.datasets["train"],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor
            )

            # Restore state if we loaded from checkpoint
            if self._train_dataloader_state is not None:
                print("[DataModule] Restoring StatefulDataLoader state")
                try:
                    dataloader.load_state_dict(self._train_dataloader_state)
                    print("[DataModule] StatefulDataLoader state restored successfully")
                except Exception as e:
                    print(f"[DataModule] Warning: Could not restore dataloader state: {e}")
                self._train_dataloader_state = None  # Clear after use

            # Store reference for state_dict() to access
            self._current_train_dataloader = dataloader

            return dataloader
        else:
            # Regular datasets use normal DataLoader
            return DataLoader(
                self.datasets["train"],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                collate_fn=custom_collate,
                pin_memory=True,
                drop_last=True,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor
            )

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate, shuffle=False,
                          pin_memory=True,
                          persistent_workers=self.persistent_workers,
                          prefetch_factor=self.prefetch_factor)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate, shuffle=False,
                          pin_memory=True,
                          persistent_workers=self.persistent_workers,
                          prefetch_factor=self.prefetch_factor)

def main():
    cli = LightningCLI(
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
