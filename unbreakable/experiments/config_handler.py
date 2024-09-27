import yaml
from pathlib import Path
from typing import Dict, Any, Union, List, Set


def load_config(country: str) -> Dict[str, Any]:
    """
    Load and validate configuration settings for the analysis.

    Args:
        country (str): Country name.

    Returns:
        Dict[str, Any]: Validated configuration settings.
    """
    config_path = Path(f"../config/{country}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file for {country} not found at {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate the configuration parameters."""
    required_params = {
        "constants": {
            "country": None,
            "spatial_units": None,
            "economic_params": {
                "average_productivity",
                "income_and_expenditure_growth",
                "consumption_utility",
                "discount_rate",
            },
            "recovery_params": {
                "use_precomputed_reconstruction_rates",
                "lambda_increment",
                "max_years",
            },
            "disaster_params": {
                "impact_data_type": None,
                "disaster_impact_params": {
                    "add_income_loss",
                    "poverty_bias_factor",
                    "distribution",
                    "max_bias",
                    "min_bias",
                },
                "determine_affected_params": {
                    "loss_margin_pct",
                    "max_random_threshold",
                    "min_random_threshold",
                    "num_simulations",
                },
            },
            "income_params": {"estimate", "randomize", "distribution", "delta"},
            "savings_params": {
                "estimate",
                "randomize",
                "cap_with_max_savings_rate",
                "distribution",
                "delta",
                "max_savings_rate",
            },
            "rent_params": {
                "estimate",
                "randomize",
                "distribution",
                "pct_of_income",
                "delta",
            },
            "effective_capital_stock_params": {"estimate"},
            "dwelling_params": {"estimate"},
            "dwelling_vulnerability_params": {
                "randomize",
                "distribution",
                "low",
                "high",
                "min_threshold",
                "max_threshold",
            },
            "min_households": None,
            "atol": None,
            "analysis_params": {
                "save_households",
                "save_consumption_recovery",
                "distributional_impacts",
            },
        }
    }

    for key, sub_params in required_params.items():
        if key not in config:
            raise ValueError(f"Top-level key '{key}' not found in configuration.")

        if sub_params:
            validate_sub_params(config[key], sub_params, key)


def validate_sub_params(
    config_section: Dict[str, Any],
    required_params: Dict[str, Union[None, List, Set]],
    parent_key: str,
) -> None:
    """Validate sub-parameters of a configuration section."""
    for sub_key, expected_values in required_params.items():
        if sub_key not in config_section:
            raise ValueError(f"Sub-parameter '{sub_key}' not found in '{parent_key}'.")

        if isinstance(expected_values, (list, set)):
            validate_value(
                config_section[sub_key], expected_values, f"{parent_key}.{sub_key}"
            )


def validate_value(
    value: Any, expected_values: Union[List, Set], param_path: str
) -> None:
    """Validate a single configuration value."""
    if isinstance(expected_values, set):
        missing_keys = expected_values - set(value.keys())
        if missing_keys:
            raise ValueError(f"Missing keys {missing_keys} in {param_path}.")
    elif value not in expected_values:
        raise ValueError(
            f"Value '{value}' for {param_path} not in valid list {expected_values}."
        )


def update_config(config, disaster_spec):
    """Update the configuration settings with the disaster specification."""
    config["constants"].update(
        {
            "disaster_spec": disaster_spec,
        }
    )

    # Event time cannot be greater than years to recover
    for event in disaster_spec:
        if event["event_time"] > config["constants"]["recovery_params"]["max_years"]:
            raise ValueError(
                f"Event time {event['event_time']} is greater than years to recover {config['constants']['years_to_recover']}"
            )
    return config
