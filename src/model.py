import pandas as pd
import numpy as np
import os
import random
import unittest
import json
from components.optimize import Optimizer
from utils.read_data import Reader
from utils.write_data import Writer
from tests.test_inputs import Tester

# This file contains the model class, which is the main class of the simulation model.


class Model(Reader, Optimizer, Writer, Tester):
    def __init__(self, country: str = '',
                 state: str = '',
                 district: str = '',
                 scale: str = '',
                 scenario: dict = {},
                 policy: dict = {},
                 filepath: str = '',
                 read_parameters_from_file: bool = False,
                 print_parameters: bool = False,
                 **kwargs):
        '''Initialize the model class.'''

        # ------------------------------- Basic checks ------------------------------- #
        self.country = country
        self.state = state
        self.district = district
        self.scale = scale

        # Load modules from parent classes
        super().__init__()

        # * Can do only two countries for now
        if country not in ['India', 'Saint Lucia']:
            raise ValueError(
                f'Country {country} is not supported. Please use India or Saint Lucia.')

        # * Can do country, district or state
        if scale not in ['country', 'district', 'state']:
            raise ValueError(
                f'Scale {scale} is not supported. Please use country, district or state.')

        # Check whether the provided geographical unit name is available
        self._test_availability_of_geographical_unit(
            country=country, state=state, district=district, scale=scale)

        # ------------------------------ Read parameters ----------------------------- #
        if read_parameters_from_file:
            # Read model parameters from excel file
            self.model_parameters = self._read_model_parameters(
                country=country, state=state, district=district, filepath=filepath)
        else:
            # Read model parameters from kwargs
            self.constants = kwargs['constants']
            self.uncertainties = kwargs['uncertainties']
            self.simulation = kwargs['simulation']

            # Get constants
            self.poverty_line = self.constants['poverty_line']
            self.indigence_line = self.constants['indigence_line']
            self.saving_rate = self.constants['saving_rate']

            # Get parameters
            self.is_vulnerability_random = self.uncertainties['is_vulnerability_random']
            self.income_and_expenditure_growth = self.uncertainties['income_and_expenditure_growth']
            self.poverty_bias = self.uncertainties['poverty_bias']
            self.discount_rate = self.uncertainties['discount_rate']
            self.consumption_utility = self.uncertainties['consumption_utility']
            self.adjust_assets_and_expenditure = self.uncertainties['adjust_assets_and_expenditure']
            self.min_households = self.uncertainties['min_households']

            # Get simulation parameters
            self.n_replications = self.simulation['n_replications']
            self.optimization_timestep = self.simulation['optimization_timestep']

            self.return_period = scenario
            self.policy = policy

        # Read function parameters
        self._read_function_parameters()

        # ----------------------------- Read asset damage ---------------------------- #
        asset_damage = self._read_asset_damage()
        self.event_damage = asset_damage['event_damage']
        self.total_asset_stock = asset_damage['total_asset_stock']
        self.expected_loss_fraction = self.event_damage / self.total_asset_stock
        print('Event damage: ', '{:,}'.format(self.event_damage))
        print('Total asset stock: ', '{:,}'.format(self.total_asset_stock))
        print('Expected loss fraction: ', round(
            self.expected_loss_fraction, 2))

        # Get appropriate data file names
        f = self._get_file_name(
            country=self.country, state=self.state, district=self.district, scale=self.scale)

        # Set up results directory
        self.results_directory = f'../experiments/{self.country}/{f}/{self.policy}/{self.return_period}/'

        # Create results directory if it does not exist
        if not os.path.isdir(self.results_directory):
            os.makedirs(self.results_directory)

        # Read results of the optimization
        self.optimization_results_filename = self.results_directory + \
            f"optimization_results={int(self.income_and_expenditure_growth*1E2)}.csv"

        # ------------------------ Read household survey data ------------------------ #
        self.household_data_filename = f"../data/processed/household_survey/{self.country}/{self.country}.csv"
        self.household_column_id = 'hhid'
        self.household_data = self._read_household_data()

        # Fix random seed for reproducibility in duplicating households
        np.random.seed(0)
        self._duplicate_households()
        # Just a placeholder for now, later it is going to be filled by _set_policy_response()
        self.affected_households = None
        self.average_productivity = self._calculate_average_productivity()
        self.optimization_results = self._get_optimization_results()
        # Adjust assets and expenditure of household file to match data of asset damage file
        if self.adjust_assets_and_expenditure:
            self._adjust_assets_and_expenditure()

        # Calculate probable maximum loss
        self._calculate_pml()

        # Save household data for future analysis
        self.household_data.to_csv(
            self.results_directory + '/household_data.csv', index=False)

        # Prepare data frames to store simulation results
        self._prepare_data_frames()

        # Collect parametesr parameters into self.parameters
        self._collect_parameters()

        with open(f'{self.results_directory}/parameters.json', 'w') as fp:
            json.dump(self.parameters, fp)

        if print_parameters:
            self._print_parameters()
        print('Initialization done!')
        print()

    def run_simulation(self):
        print('Running simulation...')
        for i in range(self.n_replications):
            print(f"Running replication {i} of {self.n_replications}")

            # Fix random seeds for reproducibility
            random.seed(i)
            np.random.seed(i)
            current_replication = f'replication_{i}'

            self._assign_savings()
            self._set_vulnerability()
            self._calculate_exposure(current_replication)
            self._determine_affected()
            self._apply_policy()
            self._write_event_results(current_replication)
            self._run_optimization()
            self._integrate_wellbeing()
            self._write_household_results(current_replication)
            self._save_affected_household_data(current_replication)

        self._save_simulation_results()

        self.optimization_results.dropna().sort_index().to_csv(
            self.optimization_results_filename)
        print('Simulation done!')

    def _assign_savings(self, print_statistics=True) -> None:
        # TODO: Add docstring
        ''''''
        # * Expenditure & savings information for Saint Lucia https://www.ceicdata.com/en/saint-lucia/lending-saving-and-deposit-rates-annual/lc-savings-rate
        x = self.household_data.eval(f'aeexp*{self.saving_rate}')
        name = '_assign_savings'
        params = self.function_parameters[name]
        mean_noise_low = params['mean_noise_low']
        mean_noise_high = params['mean_noise_high']

        if params['mean_noise_distribution'] == 'uniform':
            loc = np.random.uniform(mean_noise_low, mean_noise_high)
        else:
            raise ValueError("Only uniform distribution is supported yet.")

        scale = params['noise_scale']
        size = self.household_data.shape[0]
        clip_min = params['savings_clip_min']
        clip_max = params['savings_clip_max']

        # ?: aesav - adult equivalent household savings?
        self.household_data['aesav'] = x * \
            np.random.normal(loc, scale, size).round(
                2).clip(min=clip_min, max=clip_max)
        if print_statistics:
            print('Minimum expenditure: ', round(
                self.household_data['aeexp'].min(), 2))
            print('Maximum expenditure: ', round(
                self.household_data['aeexp'].max(), 2))
            print('Average expenditure: ', round(
                self.household_data['aeexp'].mean(), 2))
            print('Minimum savings: ', round(
                self.household_data['aesav'].min(), 2))
            print('Maximum savings: ', round(
                self.household_data['aesav'].max(), 2))
            print('Average savings: ', round(
                self.household_data['aesav'].mean(), 2))

    def _set_vulnerability(self) -> None:
        name = '_set_vulnerability'
        params = self.function_parameters[name]

        # TODO: Add docstring
        # ? Is vulnerability all the same for all households?
        # If vulnerability is random, then draw from the uniform distribution
        if self.is_vulnerability_random:
            low = 0.01
            # ?: Why not 0.95? The threshold below is 0.95.
            high = 0.90
            self.household_data['v'] = np.random.uniform(
                low, high, self.household_data.shape[0])

        # If vulnerability is not random, use v_init as a starting point and add some noise
        else:
            # ?: Why 0.6 and 1.4?
            low = 0.6
            high = 1.4
            # ?: Why 0.95?
            vulnerability_threshold = 0.95
            # v - actual vulnerability
            # v_init - initial vulnerability
            self.household_data['v'] = self.household_data['v_init'] * \
                np.random.uniform(low, high, self.household_data.shape[0])

            # If vulnerability turned out to be (drawn) is above the threhold, set it to the threhold
            self.household_data.loc[self.household_data['v']
                                    > vulnerability_threshold, 'v'] = vulnerability_threshold

    def _calculate_exposure(self, current_replication: str) -> None:
        # TODO: Add docstring
        # TODO: Do not hard code the poverty parameters here. Move it to a config file.

        # Random value for poverty bias
        if self.poverty_bias == 'random':
            # ?: Why 0.5 and 1.5?
            lower = 0.5
            upper = 1.5
            povbias = np.random.uniform(lower, upper)
        else:
            povbias = self.poverty_bias

        # Store the poverty bias in the simulation parameters
        # TODO: Change the name of the file to something more meaningful
        self.simulation_parameters.loc[current_replication,
                                       'poverty_bias'] = povbias

        # ?: What is fa?
        # !: Why 1?
        self.household_data['poverty_bias'] = 1
        self.household_data.loc[self.household_data['is_poor']
                                == True, 'poverty_bias'] = povbias
        # ?: What is fa0?
        delimeter = self.household_data[['keff', 'v', 'poverty_bias', 'popwgt']].prod(
            axis=1).sum()

        fa0 = self.pml / delimeter

        # !: Double multiplication?
        self.household_data['fa'] = fa0*self.household_data[['poverty_bias']]

        # !: self.household_data['fa'] seems to be the same for all households
        self.household_data.drop('poverty_bias', axis=1, inplace=True)

    def _determine_affected(self) -> None:
        # TODO: Add docstring
        # TODO: Do not hard code the parameters here. Move it to a config file.
        lower = 0
        upper = 1

        # fa - fraction affected

        # !: This is very random
        self.household_data['affected'] = self.household_data['fa'] >= np.random.uniform(
            lower, upper, self.household_data.shape[0])

        print('Number of affected households:',
              self.household_data['affected'].sum())

        # ? What does it mean?
        # TODO: Create model construction with bifurcate option.
        # Instead of selecting random number of household just make a fraction of them affected

    def _apply_policy(self) -> None:
        # TODO: Add docstring
        # TODO: Do not hard code the parameters here. Move it to a config file.
        # TODO: Comment on what policies mean
        self.household_data['DRM_cost'] = 0
        self.household_data['DRM_cash'] = 0

        if self.policy != 'None':
            print('Applying policy:', self.policy)

        if self.policy == 'Existing_SP_100':
            beneficiaries = self.household_data['affected'] == True
            # accounting
            self.household_data.loc[beneficiaries,
                                    'DRM_cost'] = self.household_data.loc[beneficiaries, 'aesoc']
            self.household_data['DRM_cost'] = self.household_data['DRM_cost'].fillna(
                0)
            self.household_data.loc[beneficiaries,
                                    'DRM_cash'] = self.household_data.loc[beneficiaries, 'aesoc']
            self.household_data['DRM_cash'] = self.household_data['DRM_cash'].fillna(
                0)
            # effect
            self.household_data.loc[beneficiaries,
                                    'aesav'] += self.household_data.loc[beneficiaries, 'aesoc']

        elif self.policy == 'Existing_SP_50':
            beneficiaries = self.household_data['affected'] == True
            # accounting
            self.household_data.loc[beneficiaries,
                                    'DRM_cost'] = self.household_data.loc[beneficiaries, 'aesoc'] * 0.5
            self.household_data['DRM_cost'] = self.household_data['DRM_cost'].fillna(
                0)
            self.household_data.loc[beneficiaries,
                                    'DRM_cash'] = self.household_data.loc[beneficiaries, 'aesoc'] * 0.5
            self.household_data['DRM_cash'] = self.household_data['DRM_cash'].fillna(
                0)
            # effect
            self.household_data.loc[beneficiaries,
                                    'aesav'] += self.household_data.loc[beneficiaries, 'aesoc'] * 0.5

        elif self.policy == 'retrofit':
            # accounting
            self.household_data['DRM_cost'] = 0.05*self.household_data[['keff', 'aewgt']
                                                                       ].prod(axis=1) * ((self.household_data['v']-0.70)/0.2).clip(lower=0.)
            self.household_data['DRM_cash'] = 0
            # effect
            self.household_data['v'] = self.household_data['v'].clip(upper=0.7)

        elif self.policy == 'retrofit_roof1':
            beneficiaries = self.household_data['roof_material'].isin([
                                                                      2, 4, 5, 6])
            # accounting
            self.household_data.loc[beneficiaries, 'DRM_cost'] = 0.05 * \
                self.household_data['keff'] * (0.1/0.2)
            self.household_data.loc[beneficiaries, 'DRM_cash'] = 0
            # effect
            self.household_data.loc[beneficiaries, 'v'] -= 0.1

        elif self.policy == 'PDS':
            # &((self.household_data['aeexp']<=self.household_data['vul_line'])|(self.household_data['age']>=65)|(self.household_data['cct_ae']>0)|(self.household_data['uct_ae']>0))
            beneficiaries = (self.household_data['affected'] == True) & (
                self.household_data['own_rent'] == 'own')
            # accounting
            self.household_data.loc[beneficiaries, 'DRM_cost'] = self.household_data.loc[beneficiaries].eval(
                'keff*v')
            self.household_data['DRM_cost'] = self.household_data['DRM_cost'].fillna(
                0)
            self.household_data.loc[beneficiaries, 'DRM_cash'] = self.household_data.loc[beneficiaries].eval(
                'keff*v')
            self.household_data['DRM_cash'] = self.household_data['DRM_cash'].fillna(
                0)

            # effect
            self.household_data.loc[beneficiaries,
                                    'aesav'] += self.household_data.loc[beneficiaries].eval('keff*v')

        else:
            # accounting
            self.household_data['DRM_cost'] = 0
            self.household_data['DRM_cash'] = 0
            # no effect

        try:
            self.affected_households = self.household_data.loc[self.household_data['affected'], ['hhid', 'hhid_original', 'popwgt', 'own_rent', 'quintile',
                                                                                                 'aeexp', 'aeexp_house', 'keff', 'v', 'aesav', 'aesoc', 'delta_tax_safety']].copy()
        except:
            self.affected_households = self.household_data.loc[self.household_data['affected'], ['hhid', 'popwgt', 'own_rent', 'quintile',
                                                                                                 'aeexp', 'aeexp_house', 'keff', 'v', 'aesav', 'aesoc', 'delta_tax_safety']].copy()
