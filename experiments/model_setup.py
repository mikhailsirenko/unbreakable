from ema_workbench import Model
from ema_workbench.em_framework.parameters import IntegerParameter, CategoricalParameter, Constant
from ema_workbench.em_framework.outcomes import ArrayOutcome
from unbreakable.model import model


def setup_model(config: dict) -> Model:
    """
    Set up the EMA Workbench model based on the provided configuration.

    Args:
        config (dict): Configuration dictionary loaded from the YAML file.

    Returns:
        Model: Configured EMA Workbench model.
    """
    my_model = Model(name="model", function=model)

    # Extract and set up uncertainties, constants, and levers from the config
    # uncertainties = config.get("uncertainties", {})
    constants = config.get("constants", {})
    levers = config.get("levers", {})

    # Define seed as an uncertainty for multiple runs,
    # By specifying a wider range, we want to ensure that the seed is likely to be different for each run
    seed_start = 0
    seed_end = 1000000000

    # Fix seed to ensure reproducibility
    # NOTE: If running multiple instances of the model in parallel, the seed will be the same for all instances
    # np.random.seed(42)

    my_model.uncertainties = [IntegerParameter(
        "random_seed", seed_start, seed_end)]

    # Constants
    my_model.constants = [Constant(key, value)
                          for key, value in constants.items()]

    # Levers
    my_model.levers = [CategoricalParameter(
        'current_policy', [values for _, values in levers.items()])]

    # Outcomes
    my_model.outcomes = [ArrayOutcome(region)
                         for region in constants.get('regions', [])]

    return my_model
