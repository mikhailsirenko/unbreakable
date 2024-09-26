from typing import Dict, Any
from ema_workbench import Model
from ema_workbench.em_framework.parameters import (
    IntegerParameter,
    CategoricalParameter,
    Constant,
)
from ema_workbench.em_framework.outcomes import ArrayOutcome
from unbreakable.model import model


def setup_model(config: Dict[str, Any]) -> Model:
    """
    Set up the EMA Workbench model based on the provided configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary loaded from the YAML file.

    Returns:
        Model: Configured EMA Workbench model.
    """
    my_model = Model(name="model", function=model)

    constants = config.get("constants", {})
    levers = config.get("levers", {})

    # Define seed as an uncertainty for multiple runs,
    # By specifying a wider range, we want to ensure that the seed is likely to be different for each run
    my_model.uncertainties = [IntegerParameter("random_seed", 0, 1_000_000_000)]
    my_model.constants = [Constant(key, value) for key, value in constants.items()]
    my_model.levers = [CategoricalParameter("current_policy", list(levers.values()))]

    # Outcomes are stored in array and calculated for each region
    my_model.outcomes = [
        ArrayOutcome(spatial_unit)
        for spatial_unit in constants.get("spatial_units", [])
    ]

    return my_model
