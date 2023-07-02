import pandas as pd


def load_experiments(districts: list,
                     policies: list,
                     scenarios: list,
                     country: str,
                     n_replications: int,
                     outcomes_of_interest: list = [
                         'consumption_loss',
                         'consumption_loss_NPV',
                         'v',
                         'c_t',
                         'c_t_unaffected',
                         'reco_rate',
                         'weeks_pov'],
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


def find_poor(households: pd.DataFrame, poverty_line: float, n_years: int) -> tuple:
    '''Get the poor at the beginning of the simulation and the poor at the end of the simulation

    Args:
        households (pd.DataFrame): Household dataframe
        poverty_line (float): Poverty line

    Returns:
        tuple: Number of poor at the beginning of the simulation, number of new poor at the end of the simulation, and the new poor dataframe
    '''
    # First, find the poor at the beginning of the simulation
    poor_initial = households[households['is_poor'] == True]
    n_poor_initial = round(poor_initial['popwgt'].sum())

    # Second, find the new poor at the end of the simulation (`n_years``)
    not_poor = households[households['is_poor'] == False]
    not_poor_affected = not_poor[not_poor['is_affected'] == True]
    x = not_poor_affected['aeexp'] - \
        not_poor_affected['consumption_loss_NPV'] / n_years
    new_poor = not_poor_affected.loc[x < poverty_line, :]
    n_new_poor = round(new_poor['popwgt'].sum())

    return n_poor_initial, n_new_poor, poor_initial, new_poor


def get_people_by_years_in_poverty(new_poor: pd.DataFrame, max_years: int) -> dict:
    '''Get the number of people in poverty for each year in poverty.

    Args:
        new_poor (pd.DataFrame): New poor dataframe
        max_years (int): Maximum number of years in poverty

    Returns:
        dict: Number of people in poverty for each year in poverty
    '''
    new_poor = new_poor.assign(years_in_poverty=new_poor['weeks_pov'] // 52)
    d = {}
    for i in range(max_years):
        d[i] = round(new_poor[new_poor['years_in_poverty'] == i]
                     ['popwgt'].sum())
    return d


def calculate_poverty_gap(poor_initial: pd.DataFrame, new_poor: pd.DataFrame, poverty_line: float) -> tuple:
    '''Calculate the poverty gap at the beginning and at the end of the simulation.

    Args:
        poor_initial (pd.DataFrame): Poor at the beginning of the simulation
        new_poor (pd.DataFrame): New poor at the end of the simulation
        poverty_line (float): Poverty line

    Returns:
        tuple: Poverty gap at the beginning and at the end of the simulation

    Raises:
        Exception: If the index is duplicated
        Exception: If the poverty gap is greater than 1
    '''
    all_poor = pd.concat([poor_initial, new_poor])

    # Check whether poor initial and new poor households are not the same
    if all_poor.index.duplicated().any():
        raise Exception('Index is duplicated')

    # First, get the average expenditure of the poor at the beginning of the simulation
    average_expenditure_poor_inital = (
        poor_initial['aeexp'] * poor_initial['popwgt']).sum() / poor_initial['popwgt'].sum()
    initial_poverty_gap = (
        poverty_line - average_expenditure_poor_inital) / poverty_line

    # Now, get the average expenditure of the poor at the end of the simulation
    average_expenditure_poor_new = (
        all_poor['aeexp'] * all_poor['popwgt']).sum() / all_poor['popwgt'].sum()
    new_poverty_gap = (
        poverty_line - average_expenditure_poor_new) / poverty_line

    # Poverty gap cannot be greater than 1
    if new_poverty_gap > 1 or initial_poverty_gap > 1:
        raise Exception('Poverty gap is greater than 1')

    return initial_poverty_gap, new_poverty_gap


def calculate_average_annual_consumption_loss(affected_households: pd.DataFrame, n_years: int) -> tuple:
    '''Get the average annual consumption loss and the average annual consumption loss as a percentage of average annual consumption.

    Args:
        affected_households (pd.DataFrame): Affected households dataframe
        n_years (int): Number of years of the optimization algorithm

    Returns:
        tuple: Average annual consumption loss and average annual consumption loss as a percentage of average annual consumption

    Raises:
        Exception: If the average annual consumption loss is greater than 1
    '''
    annual_consumption_loss = (
        affected_households['consumption_loss_NPV'] / n_years * affected_households['popwgt']).sum()
    annual_average_consumption_loss = annual_consumption_loss / \
        affected_households['popwgt'].sum()
    annual_average_consumption = (
        affected_households['aeexp'] * affected_households['popwgt']).sum() / affected_households['popwgt'].sum()
    annual_average_consumption_loss_pct = annual_average_consumption_loss / \
        annual_average_consumption

    if annual_average_consumption_loss_pct > 1:
        raise Exception('Annual average consumption loss is greater than 1')

    return annual_average_consumption_loss, annual_average_consumption_loss_pct


def calculate_resilience(affected_households: pd.DataFrame, pml: float, n_resilience_more_than_1: int) -> tuple:
    '''Calculate the resilience of the affected households.

    Args:
        affected_households (pd.DataFrame): Affected households dataframe
        pml (float): Probable maximum loss

    Returns:
        tuple: Resilience and number of times resilience is greater than 1

    Raises:
        Exception: If the total consumption loss is 0
    '''
    total_consumption_loss = (
        affected_households['consumption_loss_NPV'] * affected_households['popwgt']).sum()
    if total_consumption_loss == 0:
        raise Exception('Total consumption loss is 0')

    r = pml / total_consumption_loss

    # # * Sometimes resilience is greater than 1
    # We will set it to 1 then
    if r > 5:
        r = 1
        # raise Exception('Resilience is greater than 1')
        n_resilience_more_than_1 += 1
        # continue
    return r, n_resilience_more_than_1
