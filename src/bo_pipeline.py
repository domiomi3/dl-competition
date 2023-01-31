import neps


def get_pipeline_space():
    """
    Define a hyperparameter search-space.
    :return: pipeline space dictionary (dict)
    """
    pipeline_space = dict(
        learning_rate=neps.FloatParameter(lower=1e-3, upper=0.1, log=True),
        weight_decay=neps.FloatParameter(lower=1e-5, upper=1e-2, log=True),
        momentum=neps.FloatParameter(lower=0, upper=0.8),
        dropout=neps.FloatParameter(lower=0.0, upper=0.5),
        cutmix_prob=neps.FloatParameter(lower=0.5, upper=0.9),
        beta=neps.FloatParameter(lower=0, upper=0.5)
    )
    return pipeline_space
