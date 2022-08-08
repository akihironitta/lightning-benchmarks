import numpy as np

from benchmarks.common_models import (
    BoringParityModule,
    CIFARParityModule,
    MNISTParityModule,
)
from benchmarks.common_utils import measure_loops


class BoringBenchmark:
    timeout = 10 * 60

    def setup_cache(self):
        return measure_loops(BoringParityModule, kind="PT Lightning", num_epochs=4)

    def track_time(self, output):
        return np.mean(output["durations"])

    def track_peakmem(self, output):
        return np.mean(output["memory"]) / 1024  # in KiB

    def track_last_loss(self, output):
        return np.mean(output["losses"])


class MNISTBenchmark:
    timeout = 10 * 60

    def setup_cache(self):
        return measure_loops(MNISTParityModule, kind="PT Lightning", num_epochs=4)

    def track_time(self, output):
        return np.mean(output["durations"])

    def track_peakmem(self, output):
        return np.mean(output["memory"]) / 1024  # in KiB

    def track_last_loss(self, output):
        return np.mean(output["losses"])


class CIFARBenchmark:
    timeout = 10 * 60

    def setup_cache(self):
        return measure_loops(CIFARParityModule, kind="PT Lightning", num_epochs=4)

    def track_time(self, output):
        return np.mean(output["durations"])

    def track_peakmem(self, output):
        return np.mean(output["memory"]) / 1024  # in KiB

    def track_last_loss(self, output):
        return np.mean(output["losses"])
