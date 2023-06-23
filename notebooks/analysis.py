import pandas as pd


def load_experiments(districts: list, policies: list, scenarios: list, country: str, n_replications: int,
                  outcomes_of_interest: list = ['consumption_loss', 'consumption_loss_NPV', 'v', 'c_t', 'c_t_unaffected', 'reco_rate'], 
                  print_statistics: bool = True, tests: bool = True) -> dict:
    # Outcomes structure: district -> policy -> scenario -> outcome -> replication -> data
    outcomes = {}
    for district in districts:
        outcomes[district] = {}
        for policy in policies:
            outcomes[district][policy] = {}
            for scenario in scenarios:
                outcomes[district][policy][scenario] = {}
                for replication in range(n_replications):
                    path = f'../experiments/{country}/{district}/{policy}/{scenario}/'
                    # Load households and affected_households dataframes
                    households = pd.read_feather(
                        path + f'households_{replication}.feather')
                    households.set_index('hhid', inplace=True)
                    affected_households = pd.read_feather(
                        path + f'affected_households_{replication}.feather')
                    # Keep ids of affected households
                    index = affected_households['hhid']
                    for column in outcomes_of_interest:
                        # Add outcomes of interest to households dataframe
                        # Use index of affected households to match rows
                        households.loc[index,
                                       column] = affected_households[column].values
                        households[column] = households[column].fillna(0)
                    # Move hhid to column
                    households['hhid'] = households.index
                    households.reset_index(drop=True, inplace=True)

                    # Test if `aeexp` is smaller than `consumption_loss_NPV` / `n_years`
                    test_aeexp(households)

                    outcomes[district][policy][scenario][replication] = households
        if print_statistics:
            print(f"{district}'s PML: {households['pml'].iloc[0]:,.0f}")
    return outcomes


def test_aeexp(data: pd.DataFrame, n_years: int = 10) -> None:
    '''Test if `aeexp` is smaller than `consumption_loss_NPV` / `n_years`

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with columns aeexp and consumption_loss_NPV
    n_years : int, optional
        Number of years, by default 10

    Raises
    ------
    Exception
        If `aeexp` is smaller than `consumption_loss_NPV` / `n_years`

    Returns
    -------
    None
    '''
    if ((data['aeexp'] - data['consumption_loss_NPV'] / n_years) < 0).sum() > 0:
        raise Exception(
            f'aeexp is smaller than consumption_loss_NPV / {n_years}')
