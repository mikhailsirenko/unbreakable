import pandas as pd
import numpy as np
from typing import Optional, Literal


def randomize_rent(
    households: pd.DataFrame, params: dict, random_seed: int
) -> pd.DataFrame:
    """
    Randomize household rent based on estimated rent and provided parameters.

    Args:
        households (pd.DataFrame): DataFrame containing household data with existing rent.
        params (dict): Dictionary of parameters for randomization.

    Returns:
        pd.DataFrame: DataFrame with randomized rent.

    Raises:
        ValueError: If 'rent' column is not present in the DataFrame.
    """
    np.random.seed(random_seed)
    if "rent" not in households.columns:
        raise ValueError("'rent' column not found. Please estimate rent first.")

    households = households.copy()
    delta = params.get("delta", 0.05)
    distribution = params.get("distribution", "uniform")

    if distribution == "uniform":
        multiplier = 1 + np.random.uniform(-delta, delta, len(households))
    else:
        raise ValueError(
            f"Distribution '{distribution}' is not supported. Currently, only 'uniform' is available."
        )

    households["rent"] = households["rent"] * multiplier
    return households


def randomize_savings(
    households: pd.DataFrame, params: dict, random_seed: int
) -> pd.DataFrame:
    """
    Randomize household savings based on estimated savings and provided parameters.

    Args:
        households (pd.DataFrame): DataFrame containing household data with existing savings.
        params (dict): Dictionary of parameters for randomization.
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with randomized savings.

    Raises:
        ValueError: If 'sav' or 'inc' columns are not present in the DataFrame.
    """
    if "sav" not in households.columns or "inc" not in households.columns:
        raise ValueError(
            "Both 'sav' and 'inc' columns must be present to randomize savings."
        )
    np.random.seed(random_seed)

    households = households.copy()
    saving_rate = households["sav"] / households["inc"]
    delta = params.get("delta", 0.1)
    distribution = params.get("distribution", "uniform")

    if distribution == "uniform":
        multiplier = 1 + np.random.uniform(-delta, delta, len(households))
    else:
        raise ValueError(
            f"Distribution '{distribution}' is not supported. Currently, only 'uniform' is available."
        )

    households["sav"] = households["inc"] * saving_rate * multiplier

    max_savings_rate = params.get("max_savings_rate", 0.2)
    households["sav"] = np.minimum(
        households["sav"], households["inc"] * max_savings_rate
    )

    return households


def randomize_income(
    households: pd.DataFrame, params: dict, random_seed: int
) -> pd.DataFrame:
    """
    Randomize household incomes based on estimated incomes and provided parameters.

    Args:
        households (pd.DataFrame): DataFrame containing household data with estimated incomes.
        params (dict): Dictionary of parameters for randomization.
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with randomized incomes.

    Raises:
        ValueError: If 'inc' column is not present in the DataFrame.
    """
    np.random.seed(random_seed)
    if "inc" not in households.columns:
        raise ValueError("'inc' column not found. Please estimate incomes first.")

    households = households.copy()
    randomized_ratio = randomize_ratio(
        households["inc"] / households["exp"],
        size=len(households),
        distribution=params.get("distribution", "uniform"),
        delta=params.get("delta", 0.1),
    )
    households["inc"] = households["exp"] * randomized_ratio
    return households


def randomize_ratio(
    base_ratio: pd.Series, size: int, distribution: str = "uniform", delta: float = 0.1
) -> np.ndarray:
    """
    Generate randomized income-to-expenditure ratios.

    This function creates an array of randomized ratios based on a base ratio and specified distribution.

    Args:
        base_ratio (pd.Series): The base ratios to randomize.
        size (int): Number of ratios to generate.
        distribution (str): Type of distribution to use (default: 'uniform').
        delta (float): Range of variation around the base_ratio (default: 0.1).

    Returns:
        np.ndarray: Array of randomized ratios.

    Raises:
        ValueError: If an unsupported distribution is specified.
    """
    if distribution == "uniform":
        low = np.maximum(base_ratio - delta, 1.01)  # Ensure ratio doesn't go below 1.01
        high = base_ratio + delta
        return np.random.uniform(low, high, size)
    else:
        raise ValueError(
            f"Distribution '{distribution}' is not supported. Currently, only 'uniform' is available."
        )


def randomize_dwelling_vulnerability(
    households: pd.DataFrame, params: dict, random_seed: int
) -> pd.DataFrame:
    """
    Randomize dwelling vulnerability.

    Args:
        households (pd.DataFrame): Households DataFrame with 'v_init' column (initial vulnerability).
        params (dict): Dict of parameters for randomization.
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Households with randomized vulnerability ('v' column added).

    Raises:
        ValueError: If the distribution is not supported or if required parameters are missing.
    """
    if not params.get("randomize", False):
        if "v" not in households.columns:
            households["v"] = households["v_init"]
        return households

    np.random.seed(random_seed)

    distribution = params.get("distribution", "uniform")

    if distribution != "uniform":
        raise ValueError(
            f"Distribution '{distribution}' is not supported. Only 'uniform' is currently supported."
        )

    # Extract parameters with default values
    low = params.get("low", 0.9)
    high = params.get("high", 1.1)
    max_threshold = params.get("max_threshold", 0.9)
    min_threshold = params.get("min_threshold", 0.2)

    # Validate parameters
    if not (0 < min_threshold < max_threshold < 1):
        raise ValueError(
            f"Invalid 'min_thresh' and 'max_thresh' values: {min_threshold}, {max_threshold}. Must be 0 < min_thresh < max_thresh < 1."
        )

    # Generate random values and calculate new vulnerability
    random_factors = np.random.uniform(low, high, households.shape[0])
    households["v"] = households["v_init"] * random_factors

    # Apply thresholds using numpy for efficiency
    households["v"] = np.clip(households["v"], min_threshold, max_threshold)

    return households
