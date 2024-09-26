import pandas as pd
import numpy as np


def resample_households_by_spatial_unit(
    households: pd.DataFrame, min_households: int, random_seed: int = None
) -> pd.DataFrame:
    """Resample country households to be more representative.

    Args:
        households (pd.DataFrame): Households data.
        min_households (int): Minimum number of households for representative sample.
        random_seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        pd.DataFrame: Resampled households.
    """
    households = households.copy()
    households["household_id_orig"] = households["household_id"]
    households["household_weight_orig"] = households["household_weight"]

    resampled = (
        households.groupby("spatial_unit")
        .apply(
            lambda x: upsample_spatial_unit_households(x, min_households, random_seed)
        )
        .reset_index(drop=True)
    )

    resampled["household_id"] = range(1, len(resampled) + 1)
    return resampled


def upsample_spatial_unit_households(
    households: pd.DataFrame, min_households: int, random_seed: int = None
) -> pd.DataFrame:
    """Weighted resampling with adjustment for household representation within a spatial unit.

    Args:
        households (pd.DataFrame): Households of a specific spatial unit.
        min_households (int): Minimum number of households for representative sample.
        random_seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        pd.DataFrame: Resampled households of a specific spatial unit.

    Raises:
        ValueError: If total weights after resampling is not equal to initial total weights.
    """
    np.random.seed(random_seed)
    if len(households) >= min_households:
        return households

    initial_total_weights = households["household_weight"].sum()
    delta = max(min_households - len(households), 2)  # Ensure at least 2 samples

    sample = households.sample(n=delta, replace=True, random_state=random_seed)
    duplicated_households = pd.concat([households, sample], ignore_index=True)

    duplication_counts = duplicated_households.groupby("household_id").size()
    duplicated_households["household_weight"] /= duplication_counts[
        duplicated_households["household_id"]
    ].values

    if not np.isclose(
        duplicated_households["household_weight"].sum(),
        initial_total_weights,
        atol=1e-6,
    ):
        raise ValueError(
            "Total weights after duplication is not equal to the initial total weights"
        )

    return duplicated_households
