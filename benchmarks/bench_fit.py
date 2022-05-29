import numpy as np
from torch.utils.data import DataLoader

from benchmarks.common_data import RandomDataset
from benchmarks.common_models import BoringModel, ParityModuleCIFAR, ParityModuleMNIST, ParityModuleRNN
from benchmarks.common_utils import measure_loops
from pytorch_lightning import Trainer


class MNISTBenchmark:
    timeout = 10 * 60

    def setup_cache(self):
        return measure_loops(ParityModuleMNIST, kind="PT Lightning", num_epochs=4)

    def track_time(self, output):
        return np.mean(output["durations"])

    def track_peakmem(self, output):
        return np.mean(output["memory"]) / 1024  # in KiB

    def track_last_loss(self, output):
        return np.mean(output["losses"])


class RNNBenchmark:
    def setup_cache(self):
        return measure_loops(ParityModuleRNN, kind="PT Lightning", num_epochs=4)

    def track_time(self, output):
        return np.mean(output["durations"])

    def track_peakmem(self, output):
        return np.mean(output["memory"]) / 1024  # in KiB

    def track_last_loss(self, output):
        return np.mean(output["losses"])


class CIFARBenchmark:
    timeout = 10 * 60

    def setup_cache(self):
        return measure_loops(ParityModuleCIFAR, kind="PT Lightning", num_epochs=4)

    def track_time(self, output):
        return np.mean(output["durations"])

    def track_peakmem(self, output):
        return np.mean(output["memory"]) / 1024  # in KiB

    def track_last_loss(self, output):
        return np.mean(output["losses"])


class BoringBenchmark:
    def setup_cache(self):
        train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
        model = BoringModel()
        trainer = Trainer(
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            max_epochs=20,
        )
        return trainer, model, train_data

    def time_it(self, args):
        trainer, model, train_data = args
        trainer.fit(model, train_data)

    def peakmem_it(self, args):
        trainer, model, train_data = args
        trainer.fit(model, train_data)


class AnotherBoringBenchmark:
    def fit_model(self):
        data = DataLoader(RandomDataset(32, 64), batch_size=2)
        model = BoringModel()
        trainer = Trainer(
            accelerator=None,
            max_epochs=1,
            logger=False,
            benchmark=True,
        )
        trainer.fit(model, data)
        trainer.test(model, data)
        trainer.validate(model, data)

    def time_it(self):
        self.fit_model()

    def peakmem_it(self):
        self.fit_model()
