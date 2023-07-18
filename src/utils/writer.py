import pandas as pd
import numpy as np

# TODO: Make sure that you skip runs with no affected households


class Writer:
    '''Writer class for preparing model outcomes for analysis.'''

    def _prepare_outcomes(self) -> None:
        '''Add columns/outcomes of interest from `affected households` to the `households` dataframe.'''
        
        outcomes_of_interest: list = [
            'consumption_loss',
            'consumption_loss_NPV',
            'c_t',
            'c_t_unaffected',
            'recovery_rate',
            'weeks_pov']
        columns = ['hhid'] + outcomes_of_interest
        self.households = pd.merge(self.households, self.affected_households[columns], on='hhid', how='left')

    def _get_outcomes(self) -> None:
        self._prepare_outcomes()
        x_max = self.x_max
    
        households = self.households.copy()
        total_population = households['popwgt'].sum()

        # Save some outcomes for verification
        event_damage = self.event_damage
        total_asset_stock = self.total_asset_stock
        expected_loss_fraction = self.expected_loss_fraction
        average_productivity = self.average_productivity
        total_asset_in_survey = self.total_asset_in_survey

        # Actual outcomes of interest
        affected_households = households[households['is_affected'] == True]
        total_asset_loss = affected_households[['keff', 'v', 'popwgt']].prod(axis=1).sum()
        total_consumption_loss = affected_households[['consumption_loss_NPV', 'popwgt']].prod(axis=1).sum()
        n_affected_people = affected_households['popwgt'].sum()
        annual_average_consumption = (
            households['aeexp'] * households['popwgt']).sum() / households['popwgt'].sum()

        # * Poverty line is different across replications
        poverty_line = households['poverty_line_adjusted'].values[0]

        if poverty_line == 0:
            raise ValueError('Poverty line is zero')

        # Get PML, its the same across replications and stored in households
        pml = households['pml'].iloc[0]

        # * Some runs give no affected households and we will skip these
        if len(affected_households) == 0:
            # no_affected_households += 1
            pass

        # * Sometimes households are affected but they have no consumption loss
        if affected_households['consumption_loss_NPV'].sum() == 0:
            # zero_consumption_loss += 1
            pass

        n_poor_initial, n_new_poor, poor_initial, new_poor = self._find_poor(
            households, poverty_line, x_max)
        
        if poverty_line == 0:
            raise ValueError('Poverty line is zero')

        # years_in_poverty = get_people_by_years_in_poverty(new_poor, max_years)
        initial_poverty_gap, new_poverty_gap = self._calculate_poverty_gap(
            poor_initial, new_poor, poverty_line)
        
        if poverty_line == 0:
            raise ValueError('Poverty line is zero')

        annual_average_consumption_loss, annual_average_consumption_loss_pct = self._calculate_average_annual_consumption_loss(
            affected_households, x_max)
        r = self._calculate_resilience(
            affected_households, pml)

        if poverty_line == 0:
            raise ValueError('Poverty line is zero')


        return {
            'total_population': total_population,
            'total_asset_loss': total_asset_loss,
            'total_consumption_loss': total_consumption_loss,
            'event_damage' : event_damage,
            'total_asset_stock' : total_asset_stock,
            'average_productivity' : average_productivity,
            'total_asset_in_survey' : total_asset_in_survey,
            'expected_loss_fraction' : expected_loss_fraction,
            'n_affected_people': n_affected_people,
            'annual_average_consumption': annual_average_consumption,
            'poverty_line': poverty_line,
            'pml': pml,
            'n_poor_initial': n_poor_initial,
            'n_new_poor': n_new_poor,
            'initial_poverty_gap': initial_poverty_gap,
            'new_poverty_gap': new_poverty_gap,
            'annual_average_consumption_loss': annual_average_consumption_loss,
            'annual_average_consumption_loss_pct': annual_average_consumption_loss_pct,
            'r': r,
            # 'n_resilience_more_than_1' : n_resilience_more_than_1
        }

    def _find_poor(self, households: pd.DataFrame, poverty_line: float, x_max: int) -> tuple:
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

        # Second, find the new poor at the end of the simulation (`x_max``)
        not_poor = households[households['is_poor'] == False]
        
        # if poor_initial['popwgt'].sum() + not_poor['popwgt'].sum() != households['popwgt'].sum():
        #     raise ValueError('Poor and not poor do not add up to total population')
        
        not_poor_affected = not_poor[not_poor['is_affected'] == True]
        x = not_poor_affected['aeexp'] - \
            not_poor_affected['consumption_loss_NPV'] / x_max
        new_poor = not_poor_affected.loc[x < poverty_line, :]
        n_new_poor = round(new_poor['popwgt'].sum())

        return n_poor_initial, n_new_poor, poor_initial, new_poor

    def _get_people_by_years_in_poverty(self, new_poor: pd.DataFrame, max_years: int) -> dict:
        '''Get the number of people in poverty for each year in poverty.

        Args:
            new_poor (pd.DataFrame): New poor dataframe
            max_years (int): Maximum number of years in poverty

        Returns:
            dict: Number of people in poverty for each year in poverty
        '''
        new_poor = new_poor.assign(
            years_in_poverty=new_poor['weeks_pov'] // 52)
        d = {}
        for i in range(max_years):
            d[i] = round(new_poor[new_poor['years_in_poverty'] == i]
                         ['popwgt'].sum())
        return d

    def _calculate_poverty_gap(self, poor_initial: pd.DataFrame, new_poor: pd.DataFrame, poverty_line: float) -> tuple:
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
        # First, get the average expenditure of the poor at the beginning of the simulation
        average_expenditure_poor_inital = (
            poor_initial['aeexp'] * poor_initial['popwgt']).sum() / poor_initial['popwgt'].sum()
        initial_poverty_gap = (
            poverty_line - average_expenditure_poor_inital) / poverty_line

        all_poor = pd.concat([poor_initial, new_poor])

        # Check whether poor initial and new poor households are not the same
        if all_poor.index.duplicated().any():
            raise Exception('Index is duplicated')

        # Now, get the average expenditure of the poor at the end of the simulation
        average_expenditure_poor_new = (
            all_poor['aeexp'] * all_poor['popwgt']).sum() / all_poor['popwgt'].sum()
        new_poverty_gap = (
            poverty_line - average_expenditure_poor_new) / poverty_line

        # Poverty gap cannot be greater than 1
        if new_poverty_gap > 1 or initial_poverty_gap > 1:
            raise Exception('Poverty gap is greater than 1')

        return initial_poverty_gap, new_poverty_gap

    def _calculate_average_annual_consumption_loss(self, affected_households: pd.DataFrame, x_max: int) -> tuple:
        '''Get the average annual consumption loss and the average annual consumption loss as a percentage of average annual consumption.

        Args:
            affected_households (pd.DataFrame): Affected households dataframe
            x_max (int): Number of years of the optimization algorithm

        Returns:
            tuple: Average annual consumption loss and average annual consumption loss as a percentage of average annual consumption

        Raises:
            Exception: If the average annual consumption loss is greater than 1
        '''
        
        if len(affected_households) == 0:
            return np.nan, np.nan
        
        annual_consumption_loss = (
            affected_households['consumption_loss_NPV'] / x_max * affected_households['popwgt']).sum()

        annual_average_consumption_loss = annual_consumption_loss / \
            affected_households['popwgt'].sum()

        annual_average_consumption = (
            affected_households['aeexp'] * affected_households['popwgt']).sum() / \
                affected_households['popwgt'].sum()

        annual_average_consumption_loss_pct = annual_average_consumption_loss / \
            annual_average_consumption

        if annual_average_consumption_loss_pct > 1:
            raise Exception(
                'Annual average consumption loss is greater than 1')

        return annual_average_consumption_loss, annual_average_consumption_loss_pct

    def _calculate_resilience(self, affected_households: pd.DataFrame, pml: float,
                              # n_resilience_more_than_1: int
                              ) -> tuple:
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
            # raise Exception('Total consumption loss is 0')
            r = np.nan
        else:
            r = pml / total_consumption_loss

        # !: Sometimes resilience is greater than 1
        # We will set it to 1 then
        if r > 5:
            r = 1
            # raise Exception('Resilience is greater than 1')
            # n_resilience_more_than_1 += 1
            # continue
        return r