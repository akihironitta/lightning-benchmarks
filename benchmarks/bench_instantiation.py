import torch

from pytorch_lightning import Trainer


class TrainerInstantiationSuite:
    param_names = ["accelerator"]
    params = [["cpu", "gpu"]]

    def setup(self, accelerator):
        if accelerator == "gpu" and not torch.cuda.is_available():
            raise NotImplementedError

    def time_it(self, accelerator):
        Trainer(accelerator=accelerator, devices=1)
