import pandas as pd
import numpy as np
import random
from utils.reader import Reader
from optimize.optimizer import Optimizer
from utils.writer import Writer
import pickle

# This file contains the model class, which is the main class of the simulation model.


class SimulationModel(Reader, Optimizer, Writer):
    def __init__(self,
                 country: str = '',
                 scale: str = '',
                 districts: list = [],
                 print_statistics: bool = False,
                 **kwargs):
        '''Initialize the model class.'''

        self.country = country
        self.scale = scale
        self.districts = districts
        self.print_statistics = print_statistics

        # Load parameters from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Load modules from parent classes
        super().__init__()

        # * Can do only one country for now
        if country not in ['Saint Lucia']:
            raise ValueError(
                f'Country {country} is not supported. Please use Saint Lucia.')

        # * Can do country only district yet
        if scale not in ['district']:
            raise ValueError(
                f'Scale {scale} is not supported. Please use district.')

        self._read_household_survey() # V
        self._duplicate_households() # V
        self._read_asset_damage() # V

    def run_model(self,
                  random_seed: int,
                  # * Use this to run the model with uncertainties
                  # poverty_bias: float,
                  # consumption_utility: float,
                  # discount_rate: float,
                  # income_and_expenditure_growth: float,
                  my_policy: str = 'None'):
        '''Run the model.'''
        random.seed(random_seed)
        np.random.seed(random_seed)

        # * Use this to run the model without uncertainties
        poverty_bias = self.poverty_bias
        consumption_utility = self.consumption_utility
        discount_rate = self.discount_rate
        income_and_expenditure_growth = self.income_and_expenditure_growth

        outcomes = {}
        for district in self.districts:
            self._get_asset_damage(district) # V
            self._prepare_households(district) # V
            self._calculate_pml() # V
            self._assign_savings() # V
            self._set_vulnerability()
            self._calculate_exposure(poverty_bias)
            self._determine_affected()
            self._apply_individual_policy(my_policy)
            self._run_optimization(consumption_utility, discount_rate)
            self._integrate_wellbeing(
                consumption_utility, discount_rate, income_and_expenditure_growth)

            # * Use this to get the household consumption recovery (per time step)
            # household_consumption = self._integrate_wellbeing(
            #     consumption_utility, discount_rate, income_and_expenditure_growth)

            # Save household_consumption as a pickle file
            # with open(f'../results/consumption/{district}.pkl', 'wb') as f:
            #     pickle.dump(household_consumption, f)

            array_outcomes = np.array(list(self._get_outcomes().values()))
            outcomes[district] = array_outcomes

        return outcomes

    def _assign_savings(self) -> None:
        '''Assign savings to households.

        We assume that savings are a product of expenditure and saving rate with Gaussian noise.
        '''
        # * Expenditure & savings information for Saint Lucia https://www.ceicdata.com/en/saint-lucia/lending-saving-and-deposit-rates-annual/lc-savings-rate

        # Savings are a product of expenditure and saving rate
        x = self.households.eval(f'aeexp*{self.saving_rate}')
        params = self._assign_savings_params

        # Get the mean of the noise with uniform distribution
        mean_noise_low = params['mean_noise_low']  # default 0
        mean_noise_high = params['mean_noise_high']  # default 5
        if params['mean_noise_distribution'] == 'uniform':
            loc = np.random.uniform(mean_noise_low, mean_noise_high)
        else:
            raise ValueError("Only uniform distribution is supported yet.")

        # Get the scale
        scale = params['noise_scale']  # default 2.5
        size = self.households.shape[0]
        clip_min = params['savings_clip_min']  # default 0.1
        clip_max = params['savings_clip_max']  # default 1.0

        # Calculate savings with normal noise
        # !: aesav can go to 0 and above 1 because of the mean noise and loc
        # !: See `verifcation.ipynb` for more details
        if params['noise_distribution'] == 'normal':
            self.households['aesav'] = x * \
                np.random.normal(loc, scale, size).round(
                    2).clip(min=clip_min, max=clip_max)
        else:
            ValueError("Only normal distribution is supported yet.")

        if self.print_statistics:
            print('Minimum expenditure: ', round(
                self.households['aeexp'].min(), 2))
            print('Maximum expenditure: ', round(
                self.households['aeexp'].max(), 2))
            print('Average expenditure: ', round(
                self.households['aeexp'].mean(), 2))
            print('Minimum savings: ', round(
                self.households['aesav'].min(), 2))
            print('Maximum savings: ', round(
                self.households['aesav'].max(), 2))
            print('Average savings: ', round(
                self.households['aesav'].mean(), 2))

    def _set_vulnerability(self) -> None:
        '''Set vulnerability of households.

        Vulnerability can be random or based on `v_init` with uniform noise. 
        '''
        params = self._set_vulnerability_params

        # If vulnerability is random, then draw from the uniform distribution
        if self.is_vulnerability_random:
            low = params['vulnerability_random_low']  # default 0.01
            high = params['vulnerability_random_high']  # default 0.90
            if params['vulnerability_random_distribution'] == 'uniform':
                self.households['v'] = np.random.uniform(
                    low, high, self.households.shape[0])
            else:
                raise ValueError(
                    "Only uniform distribution is supported yet.")

        # If vulnerability is not random, use v_init as a starting point and add some noise
        # ?: What is the point of adding the noise to the v_init if we cap it anyhow
        else:
            low = params['vulnerability_initial_low']  # default 0.6
            high = params['vulnerability_initial_high']  # default 1.4
            # v - actual vulnerability
            # v_init - initial vulnerability
            if params['vulnerability_initial_distribution'] == 'uniform':
                self.households['v'] = self.households['v_init'] * \
                    np.random.uniform(low, high, self.households.shape[0])
            else:
                raise ValueError(
                    "Only uniform distribution is supported yet.")

            # default 0.95
            vulnerability_threshold = params['vulnerability_initial_threshold']

            # If vulnerability turned out to be (drawn) is above the threshold, set it to the threshold
            self.households.loc[self.households['v']
                                > vulnerability_threshold, 'v'] = vulnerability_threshold

    def _calculate_exposure(self, poverty_bias: float) -> None:
        '''Calculate exposure of households.

        Exposure is a function of poverty bias, effective captial stock, 
        vulnerability and probable maximum loss.
        '''
        params = self._calculate_exposure_params

        # Random value for poverty bias
        if poverty_bias == 'random':
            low = params['poverty_bias_random_low']  # default 0.5
            high = params['poverty_bias_random_high']  # default 1.5
            if params['poverty_bias_random_distribution'] == 'uniform':
                povbias = np.random.uniform(low, high)
            else:
                raise ValueError(
                    "Only uniform distribution is supported yet.")
        else:
            povbias = poverty_bias

        # Set poverty bias to 1 for all households
        self.households['poverty_bias'] = 1

        # Set poverty bias to povbias for poor households
        self.households.loc[self.households['is_poor']
                            == True, 'poverty_bias'] = povbias

        # DEFF: keff - effective capital stock
        delimiter = self.households[['keff', 'v', 'poverty_bias', 'popwgt']].prod(
            axis=1).sum()

        # ?: fa - fraction affected?
        fa0 = self.pml / delimiter

        # Print delimiter and fa0 with commas for thousands
        if self.print_statistics:
            print('PML: ', '{:,}'.format(round(self.pml, 2)))
            print('Delimiter: ', '{:,}'.format(round(delimiter, 2)))
            print('f0: ', '{:,}'.format(round(fa0, 2)))

        self.households['fa'] = fa0*self.households[['poverty_bias']]

        # !: self.households['fa'] seems to be the same for all households
        self.households.drop('poverty_bias', axis=1, inplace=True)

    def _determine_affected(self) -> None:
        '''Determine affected households.


        '''
        params = self._determine_affected_params
        low = params['low']  # default 0
        high = params['high']  # default 1

        if params['distribution'] == 'uniform':
            # !: This is very random
            self.households['is_affected'] = self.households['fa'] >= np.random.uniform(
                low, high, self.households.shape[0])
        else:
            raise ValueError("Only uniform distribution is supported yet.")

        affected_households = self.households[self.households['is_affected'] == True]
        total_asset = self.households[['keff', 'popwgt']].prod(axis=1).sum()
        total_asset_loss = affected_households[['keff', 'v', 'popwgt']].prod(axis=1).sum()

        if self.print_statistics:
            n_affected = self.households['is_affected'].multiply(
                self.households['popwgt']).sum()
            fraction_affected = n_affected / self.households['popwgt'].sum()
            print('Total number of households: ', '{:,}'.format(
                round(self.households['popwgt'].sum())))
            print('Number of affected households: ',
                  '{:,}'.format(round(n_affected)))
            print(
                f'Fraction of affected households: {round((fraction_affected * 100), 2)}%')
            print('Total asset: ', '{:,}'.format(round(total_asset)))
            print('Total asset loss: ', '{:,}'.format(round(total_asset_loss)))

        # TODO: Create model construction with bifurcate option

    def _apply_individual_policy(self, my_policy: str) -> None:
        self.households['DRM_cost'] = 0
        self.households['DRM_cash'] = 0

        if my_policy == 'None':
            self.households['DRM_cost'] = 0
            self.households['DRM_cash'] = 0

        elif my_policy == 'Existing_SP_100':
            # Beneficiaries are affected households
            beneficiaries = self.households['is_affected'] == True

            # Assign `DRM_cost`` to `aesoc` to beneficiaries, for the rest 0
            self.households.loc[beneficiaries,
                                'DRM_cost'] = self.households.loc[beneficiaries, 'aesoc']
            self.households['DRM_cost'] = self.households['DRM_cost'].fillna(0)

            # Assign `DRM_cash` to `aesoc` for beneficiaries, for the rest 0
            self.households.loc[beneficiaries,
                                'DRM_cash'] = self.households.loc[beneficiaries, 'aesoc']
            self.households['DRM_cash'] = self.households['DRM_cash'].fillna(0)

            # Increase `aesav` by `aesoc`
            self.households.loc[beneficiaries,
                                'aesav'] += self.households.loc[beneficiaries, 'aesoc']

        elif my_policy == 'Existing_SP_50':
            # Beneficiaries are those who are affected
            beneficiaries = self.households['is_affected'] == True

            # Assign `DRM_cost`` to 0.5 * `aesoc` to beneficiaries, for the rest 0
            self.households.loc[beneficiaries,
                                'DRM_cost'] = self.households.loc[beneficiaries, 'aesoc'] * 0.5
            self.households['DRM_cost'] = self.households['DRM_cost'].fillna(0)

            # Assign `DRM_cash` to 0.5 * `aesoc` to beneficiaries, for the rest 0
            self.households.loc[beneficiaries,
                                'DRM_cash'] = self.households.loc[beneficiaries, 'aesoc'] * 0.5
            self.households['DRM_cash'] = self.households['DRM_cash'].fillna(0)

            # Increase `aesav` by 0.5 `aesoc`
            self.households.loc[beneficiaries,
                                'aesav'] += self.households.loc[beneficiaries, 'aesoc'] * 0.5

        elif my_policy == 'retrofit':
            params = self._apply_individual_policy_params
            a = params['retrofit_a']  # default 0.05
            b = params['retrofit_b']  # default 0.7
            c = params['retrofit_c']  # default 0.2
            clip_lower = params['retrofit_clip_lower']  # default 0
            clip_upper = params['retrofit_clip_upper']  # default 0.7
            self.households['DRM_cost'] = a * self.households[['keff', 'aewgt']
                                                              ].prod(axis=1) * ((self.households['v'] - b) / c).clip(lower=clip_lower)
            self.households['DRM_cash'] = 0
            self.households['v'] = self.households['v'].clip(upper=clip_upper)

        elif my_policy == 'retrofit_roof1':
            params = self._apply_individual_policy_params
            # default [2, 4, 5, 6]
            roof_material_of_interest = params['retrofit_roof1_roof_materials_of_interest']
            # Beneficiaries are those who have roof of a certain material
            beneficiaries = self.households['roof_material'].isin(
                roof_material_of_interest)

            a = params['retrofit_roof1_a']  # default 0.05
            b = params['retrofit_roof1_b']  # default 0.1
            c = params['retrofit_roof1_c']  # default 0.2
            d = params['retrofit_roof1_d']  # default 0.1

            self.households.loc[beneficiaries, 'DRM_cost'] = a * \
                self.households['keff'] * (b / c)

            self.households.loc[beneficiaries, 'DRM_cash'] = 0

            # Decrease vulnerability `v` by `d`
            self.households.loc[beneficiaries, 'v'] -= d

        elif my_policy == 'PDS':
            # Benefiaries are those who are affected and have their own house
            beneficiaries = (self.households['is_affected'] == True) & (
                self.households['own_rent'] == 'own')
            # accounting
            self.households.loc[beneficiaries, 'DRM_cost'] = self.households.loc[beneficiaries].eval(
                'keff*v')
            self.households['DRM_cost'] = self.households['DRM_cost'].fillna(
                0)
            self.households.loc[beneficiaries, 'DRM_cash'] = self.households.loc[beneficiaries].eval(
                'keff*v')
            self.households['DRM_cash'] = self.households['DRM_cash'].fillna(
                0)

            # Increase `aesav` by `keff*v`
            self.households.loc[beneficiaries,
                                'aesav'] += self.households.loc[beneficiaries].eval('keff*v')

        else:
            raise ValueError(
                'Policy not found. Please use one of the following: None, Existing_SP_100, Existing_SP_50, retrofit, retrofit_roof1 or PDS.')

        columns_of_interest = ['hhid',
                               'popwgt',
                               'own_rent',
                               'quintile',
                               'aeexp',
                               'aeexp_house',
                               'keff',
                               'v',
                               'aesav',
                               'aesoc',
                               'delta_tax_safety']

        self.affected_households = self.households.loc[self.households['is_affected'], columns_of_interest].copy(
        )
