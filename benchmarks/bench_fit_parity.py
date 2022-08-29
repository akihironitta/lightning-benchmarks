import numpy as np

from benchmarks.common_models import CIFARParityModule, MNISTParityModule, BoringParityModule
from benchmarks.common_utils import measure_loops


class BoringParityBenchmark:
    param_names = []

    def setup_cache(self):
        num_epochs = 4
        num_runs = 3
        pl_output = measure_loops(MNISTParityModule, kind="PT Lightning", num_epochs=num_epochs, num_runs=num_runs)
        pt_output = measure_loops(MNISTParityModule, kind="Vanilla PT", num_epochs=num_epochs, num_runs=num_runs)
        return pl_output, pt_output, num_epochs

    def track_time_diff(self, outputs):
        pl_output, pt_output, num_epochs = outputs
        # drop the first run for initialize dataset (download & filter)
        diffs = np.asarray(pl_output["durations"][1:]) - np.mean(pt_output["durations"][1:])
        diffs = diffs / num_epochs  # norm by event count
        return np.mean(diffs)

    def track_loss_diff(self, outputs):
        pl_output, pt_output, _ = outputs
        diff = 0.0
        for pl_loss, pt_loss in zip(pl_output["losses"], pt_output["losses"]):
            diff += abs(pl_loss - pt_loss)
        return diff

    def track_peakmem_diff(self, outputs):
        pl_output, pt_output, _ = outputs
        diffs = np.asarray(pl_output["memory"]) - np.mean(pt_output["memory"])
        # relative to mean reference value
        diffs = diffs / np.mean(pt_output["memory"])
        return np.mean(diffs)


class MNISTParityBenchmark:
    param_names = []

    def setup_cache(self):
        num_epochs = 4
        num_runs = 3
        pl_output = measure_loops(MNISTParityModule, kind="PT Lightning", num_epochs=num_epochs, num_runs=num_runs)
        pt_output = measure_loops(MNISTParityModule, kind="Vanilla PT", num_epochs=num_epochs, num_runs=num_runs)
        return pl_output, pt_output, num_epochs

    def track_time_diff(self, outputs):
        pl_output, pt_output, num_epochs = outputs
        # drop the first run for initialize dataset (download & filter)
        diffs = np.asarray(pl_output["durations"][1:]) - np.mean(pt_output["durations"][1:])
        diffs = diffs / num_epochs  # norm by event count
        return np.mean(diffs)

    def track_loss_diff(self, outputs):
        pl_output, pt_output, _ = outputs
        diff = 0.0
        for pl_loss, pt_loss in zip(pl_output["losses"], pt_output["losses"]):
            diff += abs(pl_loss - pt_loss)
        return diff

    def track_peakmem_diff(self, outputs):
        pl_output, pt_output, _ = outputs
        diffs = np.asarray(pl_output["memory"]) - np.mean(pt_output["memory"])
        # relative to mean reference value
        diffs = diffs / np.mean(pt_output["memory"])
        return np.mean(diffs)


class CIFARParityBenchmark:
    def setup_cache(self):
        num_epochs = 4
        num_runs = 3
        pl_output = measure_loops(CIFARParityModule, kind="PT Lightning", num_epochs=num_epochs, num_runs=num_runs)
        pt_output = measure_loops(CIFARParityModule, kind="Vanilla PT", num_epochs=num_epochs, num_runs=num_runs)
        return pl_output, pt_output, num_epochs

    def track_time_diff(self, outputs):
        pl_output, pt_output, num_epochs = outputs
        # drop the first run for initialize dataset (download & filter)
        diffs = np.asarray(pl_output["durations"][1:]) - np.mean(pt_output["durations"][1:])
        diffs = diffs / num_epochs  # norm by event count
        return np.mean(diffs)

    def track_loss_diff(self, outputs):
        pl_output, pt_output, _ = outputs
        diff = 0.0
        for pl_loss, pt_loss in zip(pl_output["losses"], pt_output["losses"]):
            diff += abs(pl_loss - pt_loss)
        return diff

    def track_peakmem_diff(self, outputs):
        pl_output, pt_output, _ = outputs
        diffs = np.asarray(pl_output["memory"]) - np.mean(pt_output["memory"])
        # relative to mean reference value
        diffs = diffs / np.mean(pt_output["memory"])
        return np.mean(diffs)
