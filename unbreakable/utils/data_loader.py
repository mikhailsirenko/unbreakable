import pandas as pd
from typing import List
from typing import Dict, Optional, Any


def load_data(params) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load households and disaster impact data for a given country and disaster specifications.

    Args:
        params (Dict[str, Any]): Parameters dictionary containing country, disaster_params, and recovery_params

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames of household and disaster impact data.
    """
    country = params["country"]
    impact_data_type = params["disaster_params"]["impact_data_type"]
    disaster_spec = params["disaster_spec"]

    households = pd.read_csv(f"../data/processed/household_survey/{country}.csv")

    unique_disasters = set(disaster["type"] for disaster in disaster_spec)

    impact_data = []
    for disaster in unique_disasters:
        if impact_data_type == "assets":
            df = pd.read_csv(
                f"../data/processed/asset_impacts/{country}/{disaster}.csv"
            )
        elif impact_data_type == "population":
            df = pd.read_csv(
                f"../data/processed/population_impacts/{country}/{disaster}.csv"
            )
        else:
            raise ValueError(
                f"Invalid impact_data_type: {impact_data_type}. Must be 'assets' or 'population'."
            )
        impact_data.append(df)

    disaster_impacts = (
        pd.concat(impact_data, ignore_index=True)
        if len(impact_data) > 1
        else impact_data[0]
    )

    return households, disaster_impacts


def load_reconstruction_rates(params: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Load precomputed optimal reconstruction rates from a file.

    Args:
        params (Dict[str, Any]): Parameters dictionary containing recovery_params.

    Returns:
        Optional[pd.DataFrame]: Dataframe of reconstruction rates if file exists and should be used, None otherwise.
    """

    if not params["recovery_params"]["use_precomputed_reconstruction_rates"]:
        return None

    try:
        return pd.read_csv(
            f"../data/generated/{params['country']}/optimal_reconstruction_rates.csv"
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Reconstruction rates file not found for {params['country']}"
        )
    except Exception as e:
        raise RuntimeError(
            f"Error loading reconstruction rates for {params['country']}: {str(e)}"
        )
