import gc
import time

import numpy as np
import torch

from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.seed import seed_everything


def hook_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        used_memory = torch.cuda.max_memory_allocated()
    else:
        used_memory = np.nan
    return used_memory


def measure_loops(cls_model: LightningModule, kind: str, num_runs: int = 1, num_epochs: int = 10):
    """Returns an array with the last loss from each epoch for each run."""
    hist_losses = []
    hist_durations = []
    hist_memory = []

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True
    for i in range(num_runs):
        gc.collect()
        if device_type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_accumulated_memory_stats()
            torch.cuda.reset_peak_memory_stats()
        time.sleep(1)

        time_start = time.perf_counter()

        _loop = lightning_loop if kind == "PT Lightning" else vanilla_loop
        final_loss, used_memory = _loop(cls_model, idx=i, device_type=device_type, num_epochs=num_epochs)

        time_end = time.perf_counter()

        hist_losses.append(final_loss)
        hist_durations.append(time_end - time_start)
        hist_memory.append(used_memory)

    return {"losses": hist_losses, "durations": hist_durations, "memory": hist_memory}


def vanilla_loop(cls_model: torch.nn.Module, idx: int, device_type: str = "cuda", num_epochs: int = 10):
    seed_everything(idx)
    torch.backends.cudnn.deterministic = True
    device = torch.device(device_type)
    model = cls_model()
    dl = model.train_dataloader()
    optimizer = model.configure_optimizers()
    model = model.to(device)
    epoch_losses = []
    # as the first run is skipped, no need to run it long
    for _ in range(num_epochs if idx > 0 else 1):
        # run through full training set
        for j, batch in enumerate(dl):
            batch = [x.to(device) for x in batch]
            loss = model.training_step(batch, j)["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # track last epoch loss
        epoch_losses.append(loss.item())
    return epoch_losses[-1], hook_memory()


def lightning_loop(cls_model: LightningModule, idx: int, device_type: str = "cuda", num_epochs: int = 10):
    seed_everything(idx)
    torch.backends.cudnn.deterministic = True
    model = cls_model()
    trainer = Trainer(
        max_epochs=num_epochs if idx > 0 else 1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        accelerator="gpu" if device_type == "cuda" else "cpu",
        devices=1,
        logger=False,
        replace_sampler_ddp=False,
    )
    trainer.fit(model)
    return trainer.fit_loop.running_loss.last().item(), hook_memory()


class DebugTimer(Callback):
    def __init__(self):
        self.t0 = dict(
            total=0.0,
            batch=0.0,
            epoch=0.0,
        )
        self.time_results = dict(
            total=[],
            batch=[],
            epoch=[],
        )

    def setup(self, *_, **__):
        self.t0["total"] = time.monotonic()

    def on_train_epoch_start(self, *_):
        self.t0["epoch"] = time.monotonic()

    def on_train_batch_start(self, *_):
        self.t0["batch"] = time.monotonic()

    def on_train_batch_end(self, *_):
        self.time_results["batch"].append(time.monotonic() - self.t0["batch"])

    def on_train_epoch_end(self, *_):
        self.time_results["epoch"].append(time.monotonic() - self.t0["epoch"])

    def teardown(self, *_, **__):
        self.time_results["total"].append(time.monotonic() - self.t0["total"])
