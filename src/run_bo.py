import os
import shutil
import logging

import neps

from src.bo_pipeline import run_pipeline
from src.bo_pipeline import get_pipeline_space


def main():
    # set_seed(124)
    logging.basicConfig(level=logging.INFO)

    pipeline_space = get_pipeline_space()
    # if os.path.exists("results/bayesian_optimization"):
    #     shutil.rmtree("results/bayesian_optimization")
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        overwrite_working_directory=True,
        root_directory="results/bayesian_optimization",
        max_evaluations_total=20,
        searcher="bayesian_optimization",
    )
    previous_results, pending_configs = neps.status(
        "results/bayesian_optimization"
    )
    neps.plot("results/hyperband")


if __name__ == "__main__":
    main()

