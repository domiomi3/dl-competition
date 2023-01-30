import neps
import torch

from src.main import main
from src.eff_net import Inceptionv4FC
from src.data_augmentations import *


def get_pipeline_space() -> dict:
    """ Define a hyperparameter search-space.

        hyperparameters:
          lr              from 1e-6 to 1e-1 (float, log)
          optimizer       Adam or  SGD (categorical)
          batch_size          from 1 to 9 (fidelity parameter)

        Returns:
            Pipeline space dictionary
        """
    pipeline_space = dict(
        lr=neps.FloatParameter(lower=1e-6, upper=1e-1, log=True),
        # weight_decay=neps.FloatParameter(lower=1, upper=3, log=True),
        # optimizer=neps.CategoricalParameter(choices=["Adam", "SGD"]),
        # num_epochs=neps.IntegerParameter(lower=1, upper=2, is_fidelity=True)
        # batch_size=neps.IntegerParameter(lower=1, upper=64, log=True),
    )
    return pipeline_space


def run_pipeline(lr, optimizer, num_epochs):
    return main("/home/dominika/PycharmProjects/DL/dl2022-competition-dl2022-large/dataset", eval("Inceptionv4FC"),
                num_epochs=50, batch_size=16, learning_rate=lr,
                train_criterion=torch.nn.CrossEntropyLoss, model_optimizer="RMSprop",
                data_augmentations=eval("resize_to_224x224"), save_model_str=None, use_all_data_to_train=False,
                exp_name=f"eff_v2_50_16_{lr}")
