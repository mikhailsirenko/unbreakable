from sklearn.linear_model import LinearRegression
import numpy.polynomial.polynomial as poly
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# TODO: Update documentation

# Prepare Saint Lucia data as an input into simulation model.

# There are 5 inputs into the simulation model:
# * 14 parameters in `parameters.xlsx` (incl. constants, uncertainties, simulation, scenario and policy parameters);
# * 3 assest damage parameters from the assest damage file (e.g. `Saint Lucia.xlsx`);
# * N model's algorithms parameters in `algorithms_parameters.xlsx`;
# * Average capital productivity (computed based on some values of household survey data);
# * A household survey (e.g. `saint_lucia.csv`).

# Here we going to prepare the latest - the household survey.
# Each row of the data is a household and each column is an attribute.
# We need to prepare it to match the format of the simulation model.
# Here is the list of columns that we need to have:

# * `hhid` - household id,
# * `popwgt` - float: [0,inf] (?: Maybe put a cap on this?),
# * `hhsize` - household size,
# * `hhweight` - household weight,
# * `state` - state, str
# * `aeexp`- float: [0,inf] (?: Maybe put a cap on this?)
# * `hhexp` - float: [0,inf] (?: Maybe put a cap on this?)
# * `is_poor` - boolean: False or True,
# * `aeinc` - float: [0,inf] (?: Maybe put a cap on this?)
# * `aesoc` - float: [0,inf] (?: Maybe put a cap on this?)
# * `k_house_ae` - float: [0,inf (?: Maybe put a cap on this?)]
# * `v_init` - initial vulnerability, float: [0,1]
# * `inc_safetynet_frac` - float: [0,1]
# * `delta_tax_safety` - float
# * `houses_owned` - int: [0,inf] (?: Maybe put a cap on this?)
# * `own_rent` - string: "own" or "rent"
# * `aeexp_house` - float
# * `percentile` - int: [1,100]
# * `decile` - int: [1,10]
# * `quintile` - int: [1,5]

# Here is the list of what we have at the very end.
# V - we have it, X - we don't have it, ! - we must have it, ? - we don't know what it is.
# hhid V
# popwgt V
# hhsize -> hhsize_ae ?
# hhweight -> hhwgt ?
# state X
# aaexp V
# hhexp X
# ispoor V
# aeinc V
# aesoc V
# k_house_ae V
# v_init V
# inc_safetynet_frac X
# delta_tax_safety X !
# houses_owned X
# own_rent V
# aaexp_house V
# percentile X
# decile V
# quintile V

def prepare_household_survey(country: str) -> None:
    '''Prepare data for the simulation model.

    Parameters
    ----------
    country : str

    Raises
    ------
    ValueError
        If the country is not supported.

    '''
    # Data preprocessing description:
    # 0. Load raw data
    # 1. Change `parentid1` to `hhid`
    # 2. Add `is_rural` column. 0 if `urban` is URBAN, if `urban` is RURAL then 1.
    # 3. Rename `tvalassets` to `kreported`.
    # 4. Rename a set of columns.
    # 5. Calculate some household attributes. Need to figure out what they mean.
    # 6. Add `financial_inst` column which has info on bank or credit union.
    # 7. Decode income attributes.
    # 8. Calculate some income attributes. Need to figure out what they mean.
    # 9. Decode housing attributes.
    # 10. Add new housing attributes. Need to figure out what they mean.
    # 11. Add new insurance attributes. Need to figure out what they mean.
    # 12. Calculate some housing attributes. Need to figure out what they mean.
    # 13. Calculate povery attributes. Need to figure out what they mean.
    # 14. Assign vulnerability by type of house.
    # 15. Subset columns of interest.
    # 16. Check which columns do we have and which do we miss.
    # 17. Add missing columns.
    # 18. Merge districts.
    # 19. Save data.

    if country != 'Saint Lucia':
        raise ValueError('Currently only Saint Lucia is supported.')

    print_statistics = True
    data = load_data(print_statistics=print_statistics)

    # * Note that the sequence of the functions is important
    result = (start_pipeline(data)
              .pipe(add_is_rural_column, print_statistics=print_statistics)
              .pipe(rename_assets_column)
              .pipe(rename_other_columns)
              .pipe(calculate_household_attributes)
              .pipe(get_bank_or_credit_union)
              .pipe(decode_demographic_attributes)
              .pipe(decode_income_attributes)
              .pipe(calculate_income_attributes)
              .pipe(decode_housing_attributes)
              .pipe(add_housing_attributes)
              .pipe(add_insurance_attributes)
              .pipe(calculate_housing_attributes)
              .pipe(calculate_poverty_attributes)
              .pipe(assign_housing_vulnerability)
              .pipe(subset_columns)
              .pipe(check_columns)
              .pipe(add_missing_columns, missing_columns=['delta_tax_safety'])
              .pipe(merge_districts)
              )

    result.to_csv(
        f'../../data/processed/household_survey/{country}/{country}.csv ')


def load_data(print_statistics: bool = True) -> pd.DataFrame:
    """Load the raw data."""
    # Read the raw data
    # * This dataset is the combined version of the household and persons files on parentid1
    data = pd.read_csv(
        '../../data/raw/household_survey/Saint Lucia/SLCHBS2016PersonV12_Housing.csv', low_memory=False)
    data.rename(columns={'parentid1': 'hhid'}, inplace=True)

    # Set the index to the household id
    data.set_index('hhid', inplace=True)

    if print_statistics:
        print('Number of rows: ', data.shape[0])
        print('Number of columns: ', data.shape[1])
        print('Number of duplicates based on index: ',
              data.index.duplicated().sum())

    return data


def start_pipeline(data: pd.DataFrame):
    """Start the data processing pipeline."""
    return data.copy()


def add_is_rural_column(data: pd.DataFrame, print_statistics: bool = True) -> pd.DataFrame:
    """Create a new column that indicates whether the household is rural or not."""
    data['is_rural'] = 0
    data.loc[data['urban'] == 'RURAL', 'is_rural'] = 1
    if print_statistics:
        print('Number of rural households: ', data['is_rural'].sum())
        print('Number of urban households: ',
              data.shape[0] - data['is_rural'].sum())
    return data


def rename_assets_column(data: pd.DataFrame) -> pd.DataFrame:
    """Rename the assets column to be more descriptive."""
    data.rename(columns={'tvalassets': 'kreported'}, inplace=True)
    return data


def rename_other_columns(data: pd.DataFrame) -> pd.DataFrame:
    '''Rename a set of columns. See function for details.'''
    data = data.rename(columns={'DISTRICT_NAME': 'district',
                                'persons': 'hhsize',
                                'totexp.x': 'hhexp',
                                'pcexpae.x': 'aeexp',
                                'hincome': 'hhinc',
                                'WT.x': 'pwgt',
                                'food.x': 'hhexp_food'})
    return data

# ? What does it mean?
# hhwgt = WT.y from full_df with group by parentid1 or WT from House data
# pwgt = WT.x from full_df or WT from Person data

# hhsize = number of people in a household

# ? What does it mean
# ! Data doesn't have this column
# pcinc = annual consumption per head

# ! That's quite an assumption
# hhinc = pcinc * hhsize

# p4_23 is monthly income
# p4_1 is months worked
# pincome is monthly income
# pincome_oth is annual other income
# aincome is pincome * p4_1 + pincome_oth

# ! Data doesn't have this column
# hincome is the sum of aincome


def calculate_household_attributes(data: pd.DataFrame) -> pd.DataFrame:
    lower = 1
    fill_na = 1
    data['popwgt'] = data.groupby('hhid')['pwgt'].transform('sum')
    data['hhwgt'] = data['popwgt'] / data['hhsize']
    data['hhsize_ae'] = (data['hhexp'] / data['aeexp']
                         ).fillna(fill_na).clip(lower=lower)
    data['aewgt'] = data['pwgt']*(data['hhsize_ae'] / data['hhsize'])
    return data


def get_bank_or_credit_union(data: pd.DataFrame) -> pd.DataFrame:
    data['financial_inst'] = 0
    data.loc[data['p1_11__3'] == 'yes - bank', 'financial_inst'] = 1
    data.loc[data['p1_11__2'] == 'yes - bank', 'financial_inst'] = 1
    data.loc[data['p1_11__2'] == 'yes - credit union', 'financial_inst'] = 1
    data.loc[data['p1_11__1'] == 'yes - bank', 'financial_inst'] = 1
    data.loc[data['p1_11__1'] == 'yes - credit union', 'financial_inst'] = 1
    return data


def decode_demographic_attributes(data: pd.DataFrame) -> pd.DataFrame:
    '''Decode the demographic attributes.'''
    data = data.rename(columns={'p1_1': 'role',
                                'p1_2': 'sex',
                                'p1_3': 'age',
                                'p1_4': 'race',
                                'p1_5': 'religion',
                                'p1_6': 'marital_status',
                                'p1_7': 'cellphone'})
    return data


def decode_income_attributes(data: pd.DataFrame) -> pd.DataFrame:
    '''Decode the income-related attributes.'''
    data = data.rename(columns={
        # 'p1_11':'bank_account',
        'p4_1': 'months_worked',
        'inc2231002': 'other_entrepreneurial',
        'inc2331001': 'remits_intl',
        'inc2341001': 'rental_income',
        'inc2351001': 'dividends',  # other
        'inc2361001': 'interest',  # other
        'inc2361002': 'other_investment_income',  # other
        'inc2371001': 'pension_public',  # UCT
        'inc2371002': 'pension_private_LCA',  # pension private
        'inc2371003': 'pension_private_int',  # pension private
        'inc2371004': 'social_security',  # UCT
        # 'inc2381001':'annuity', # other
        'inc2381002': 'public_assistance',  # CCT
        'inc2381003': 'child_support',  # other
        'inc2391001': 'scholarships',  # other
        'inc2391002': 'financial_aid',  # other
        'inc2391003': 'alimony',  # other
        'inc2391099': 'mystery'
    })
    return data


def calculate_income_attributes(data: pd.DataFrame) -> pd.DataFrame:
    data['remits_dom'] = 0

    # Primary job income
    data['primary_income'] = data[['months_worked', 'pincome']].prod(axis=1)

    # Secondary income
    data['cct'] = data['public_assistance'].copy()
    data['uct'] = data[['pension_public', 'social_security']].sum(axis=1)
    data['remits'] = data[['remits_intl', 'remits_dom']].sum(axis=1)
    data['other_sources'] = data[['dividends', 'interest', 'child_support', 'alimony', 'financial_aid',
                                  'scholarships', 'pension_private_LCA', 'pension_private_int', 'other_investment_income', 'mystery']].sum(axis=1)
    data['secondary_income'] = data[['other_entrepreneurial', 'cct',
                                     'uct', 'remits', 'rental_income', 'other_sources']].sum(axis=1)

    # Total income
    data['total_income'] = data[[
        'primary_income', 'secondary_income']].sum(axis=1)

    return data


def decode_housing_attributes(data: pd.DataFrame) -> pd.DataFrame:
    '''Decode the housing-related attributes.'''
    data = data.rename(columns={'s2': 'own_rent',
                                # owner-occupied
                                'c1900105': 'mortgage_monthly',
                                'c1900104': 'domicile_value',
                                'c1900101': 'new_home_purchase_price',
                                'c1900103': 'new_home_mortgage_monthly',
                                'c1800501': 'rental_income_furnished',
                                'c1800502': 'rental_income_unfurnished',
                                'c1800503': 'rental_income_business',
                                'c0421101': 'imputed_rent_monthly',
                                'c1252101': 'insurance_premium',
                                # rental
                                'c0411100': 'actual_rent_monthly',
                                # condition & construction
                                's9q1': 'had_shock',
                                'h1_2': 'walls',
                                'h1_3': 'roof',
                                'h1_13': 'yr_house_built'})
    return data


def add_housing_attributes(data: pd.DataFrame) -> pd.DataFrame:
    '''Introduce new housing attributes.'''
    # ! An assumption
    data['own_rent'] = data['own_rent'].replace({'own or rent free': 'own'})
    data['home_insured'] = data['insurance_premium'] > 0
    return data


def add_insurance_attributes(data: pd.DataFrame) -> pd.DataFrame:
    # ? What does it mean?
    # National insurance corporation (unemployment)
    data['NIC_enrolled'] = data['p4_18'].isin(['employer', 'self-employed'])
    data['NIC_recipient'] = data['p4_17'].isin(['yes, from the nic'])

    # Health insurance
    data = data.rename(columns={'p2_5': 'health_insurance'})
    return data


def calculate_housing_attributes(data: pd.DataFrame) -> pd.DataFrame:
    # Predict domicile value for hh that rent
    data['k_house'] = data['domicile_value'].copy().fillna(0)
    # total rent per capita per year
    data['hhexp_house'] = 12 * data['imputed_rent_monthly'].copy()
    data['hhexp_house'].update(12 * data['actual_rent_monthly'])
    data['hhexp_house'] = data['hhexp_house'].clip(lower=0).fillna(0)

    # Urban population
    training_slc = (data['domicile_value'] > 10 * data['imputed_rent_monthly']
                    ) & (data['domicile_value'] < 1E6) & (data['is_rural'] == 0)
    urban_predictor = linear_regression(data.loc[training_slc].dropna(
        subset=['domicile_value', 'imputed_rent_monthly']), 'imputed_rent_monthly', 'domicile_value', return_model=True)

    prediction_slc = (data['own_rent'] == 'rent') & (
        data['is_rural'] == 0) & (data['actual_rent_monthly'] is not None)
    data.loc[prediction_slc, 'k_house'] = urban_predictor.predict(
        data.loc[prediction_slc, 'actual_rent_monthly'].values.reshape(-1, 1))

    # Rural population
    training_slc = (data['domicile_value'] > 10 * data['imputed_rent_monthly']
                    ) & (data['domicile_value'] < 1E6) & (data['is_rural'] == 1)
    rural_predictor = linear_regression(data.loc[training_slc].dropna(
        subset=['domicile_value', 'imputed_rent_monthly']), 'imputed_rent_monthly', 'domicile_value', return_model=True)

    prediction_slc = (data['own_rent'] == 'rent') & (
        data['is_rural'] == 1) & (data['actual_rent_monthly'] is not None)
    data.loc[prediction_slc, 'k_house'] = rural_predictor.predict(
        data.loc[prediction_slc, 'actual_rent_monthly'].values.reshape(-1, 1))

    # Correct for the households that reported unreasonably low domicile value
    prediction_slc = (data['own_rent'] == 'own') & (data['is_rural'] == 0) & (
        data['k_house'] <= 10*data['imputed_rent_monthly'])
    data.loc[prediction_slc, 'k_house'] = urban_predictor.predict(
        data.loc[prediction_slc, 'imputed_rent_monthly'].values.reshape(-1, 1))

    prediction_slc = (data['own_rent'] == 'own') & (data['is_rural'] == 1) & (
        data['k_house'] <= 10*data['imputed_rent_monthly'])
    data.loc[prediction_slc, 'k_house'] = rural_predictor.predict(
        data.loc[prediction_slc, 'imputed_rent_monthly'].values.reshape(-1, 1))

    data['k_house'] = data['k_house'].clip(lower=0).fillna(0)

    return data


def calculate_poverty_attributes(data: pd.DataFrame) -> pd.DataFrame:
    # Data has four poverty levels:
    # (1) $1.90/day = 1345 (ipline190 in dataset)
    # (2) $4.00/day = 2890,
    # (3) indigence line based on food is 2123 (indline in dataset),
    # (4) relative poverty line for food and non-food items is 6443 (povline in dataset)
    # Saint Lucia's poverty line is 1.90 * 365 = $689.7 US Dollars per year,
    # discounting using the PPP exchange rate of 1.952
    # the international poverty line for Saint Lucia is 1.90 * 1.952 * 365 = EC $1, 354 (0.7% in doc have 0.66%)
    # US $4 a day PPP is 4 * 1.952 * 365 = EC $2,890 (4.4% in doc have 4%)
    # poverty highest in Dennery and Vieux-Fort

    # Domestic lines
    # !: Do not hardcode these values
    # !: Check with Bramka
    data['pov_line'] = 6443
    data['vul_line'] = 8053.75
    data['is_poor'] = data['aeexp'] <= data['pov_line']

    # Load PMT data
    # ? What is PMT?
    # TODO: Merge PMT with data
    # !: I don't have this dataset
    # pmt = pd.read_stata(inputs + 'SLNET_16April.dta')

    # Consumption quintiles and deciles
    data = data.rename(columns={'quintile.y': 'quintile',
                                'decile.y': 'decile'})

    # print('income = {} mil. EC$'.format(round(1E-6*dfout[['aewgt','aeinc']].prod(axis=1).sum(),1)))
    # print('       = {} EC$/cap'.format(round(dfout[['aewgt','aeinc']].prod(axis=1).sum()/dfout['pwgt'].sum(),1)))

    # Individual income
    for _i in ['primary_income_ae', 'cct_ae', 'uct_ae', 'remits_ae', 'other_sources_ae']:
        data[_i] = data.groupby('hhid')[_i.replace('_ae', '')].transform(
            'sum').multiply(1/data['hhsize_ae'])

    # Household consumption
    data['imputed_rent_monthly'] = data['imputed_rent_monthly'].fillna(0)
    data['housing_service_ae'] = data.groupby(
        'hhid')['imputed_rent_monthly'].transform('mean').multiply(12./data['hhsize_ae'])
    data['aeexp_house'] = data['hhexp_house'] / data['hhsize_ae']
    data['aeexp_food'] = data['hhexp_food'] / data['hhsize_ae']
    data['aeexp_other'] = data['aeexp'] - \
        data[['aeexp_house', 'aeexp_food']].sum(axis=1)

    # sum to households
    data['aesoc'] = data['cct_ae'].copy()
    data['aeinc'] = data[['primary_income_ae', 'cct_ae', 'uct_ae',
                          'remits_ae', 'other_sources_ae', 'housing_service_ae']].sum(axis=1)

    data['k_house_ae'] = data['k_house']/data['hhsize_ae']
    return data


def assign_housing_vulnerability(data: pd.DataFrame) -> pd.DataFrame:
    # !: This is quite random!
    # TODO: Do not hard code parameters here. Move them to a config file.
    data['walls'].fillna('others', inplace=True)
    data['v_walls'] = 0.1
    data.loc[data['walls'].isin(
        ['brick/blocks', 'concrete/concrete blocks']), 'v_walls'] = 0.35
    data.loc[data['walls'].isin(['wood & concrete']), 'v_walls'] = 0.5
    data.loc[data['walls'].isin(['wood/timber']), 'v_walls'] = 0.6
    data.loc[data['walls'].isin(['plywood']), 'v_walls'] = 0.7
    data.loc[data['walls'].isin(
        ['makeshift', 'others', 'other/dont know']), 'v_walls'] = 0.8
    data['roof'].fillna('others', inplace=True)
    data['v_roof'] = 0.75
    data.loc[data['roof'].isin(
        ['sheet metal (galvanize, galvalume)']), 'v_roof'] = 0.5
    data['v_init'] = 0.5 * data['v_roof'] + 0.5 * data['v_walls']
    return data


def subset_columns(data: pd.DataFrame) -> pd.DataFrame:
    '''Subset columns of interest.'''
    columns_of_interest = ['district',
                           'is_rural',
                           'hhwgt',
                           'hhsize_ae',
                           'popwgt',
                           'aeinc',
                           'aesoc',
                           'k_house_ae',
                           'own_rent',
                           'aeexp',
                           'aeexp_house',
                           'aeexp_food',
                           'aeexp_other',
                           'is_poor',
                           'v_init',
                           'v_walls',
                           'v_roof',
                           'walls',
                           'roof',
                           'quintile',
                           'decile',
                           'primary_income_ae',
                           'cct_ae',
                           'uct_ae', 'remits_ae',
                           'other_sources_ae',
                           'housing_service_ae',
                           'pov_line',
                           'vul_line']
    result = data.loc[~data.index.duplicated(
        keep='first'), columns_of_interest]
    result['aewgt'] = data['aewgt'].groupby(level='hhid').sum()

    # Keep characteristics of head of household
    household_head_columns = ['sex',
                              'age',
                              'race',
                              'religion',
                              'marital_status',
                              'cellphone',
                              'health_insurance',
                              'home_insured']  # ,'bank_account']
    result[household_head_columns] = data.loc[data['role']
                                              == 'head', household_head_columns]
    return result


def check_columns(data: pd.DataFrame) -> pd.DataFrame:

    # These are the columns of the India case
    used_columns = [
        # 'hhid',
        'aeexp',
        'is_poor',
        'aeinc',
        'aesoc',
        'k_house_ae',
        'v_init',
        # 'delta_tax_safety',
        'own_rent',
        'aeexp_house',
    ]

    extra_columns = [
        'popwgt',  # used, but seems to be not essential, just for writing
    ]

    not_used_columns = [
        'hhsize',
        'hhweight',
        'state',
        'hhexp',
        'inc_safetynet_frac',
        'houses_owned',
        'percentile',
        'decile',
        'quintile'
    ]

    # Check whether the new data has  all columns that we need
    missing_columns = []

    for column in used_columns:
        if column not in data.columns:
            missing_columns.append(column)

    # Check what columns we have besides the ones we need from used_columns
    extra_columns = [
        column for column in data.columns if column not in used_columns]
    print(f'We have the following extra columns: {extra_columns}')

    if len(missing_columns) > 0:
        raise ValueError(f'Missing columns: {missing_columns}')

    return data


def add_missing_columns(data: pd.DataFrame, missing_columns: list) -> pd.DataFrame:
    '''Manually add missing columns to the data.'''
    for column in missing_columns:
        data[column] = 0
    return data


def merge_districts(data: pd.DataFrame) -> pd.DataFrame:
    # !: We merged two districts into one
    data['district_original'] = data['district']
    data.replace({'district': {'Castries Sub-Urban': 'Castries',
                               'Castries City': 'Castries'}}, inplace=True)
    return data


# Some regression-alike functions
# * I did not test them
np.random.seed(123)


def exponential_regression(data: pd.DataFrame, X_column: str, y_column: str, weights: np.array = None, return_model: bool = False) -> tuple[np.array, float]:
    X = data[X_column].values.reshape(-1, 1)
    y = data[y_column].values.reshape(-1, 1)
    transformer = FunctionTransformer(np.log, validate=True)
    y_transformed = transformer.fit_transform(y)

    lr = LinearRegression()
    lr.fit(X, y_transformed, sample_weight=weights)
    y_pred = lr.predict(X)
    coef = lr.coef_
    r2 = lr.score(X, y_transformed, sample_weight=weights)
    if return_model:
        return lr
    else:
        return y_pred, coef, r2


def polynomial_regression(data: pd.DataFrame,
                          X_column: str,
                          y_column: str,
                          power: int,
                          weights: np.array = None,
                          X_new: np.array = None,
                          X_start: int = 0,
                          X_end: int = 40,
                          X_num: int = 100):
    # !: Weights are not used in this function
    X = data[X_column].squeeze().T
    y = data[y_column].squeeze().T
    coef = poly.polyfit(X, y, power)

    if X_new is None:
        X_new = np.linspace(X_start, X_end, num=X_num)

    f = poly.polyval(X_new, coef)

    return X_new, f


def linear_regression(data: pd.DataFrame, X_column: str, y_column: str, weights: np.array = None, return_model: bool = False) -> tuple[np.array, float, float]:
    '''Do a linear regression on the data and return the predicted values, the coefficient and the r2 score.'''
    X = data[X_column].values.reshape(-1, 1)
    y = data[y_column].values.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(X, y, sample_weight=weights)
    y_pred = lr.predict(X)
    coef = lr.coef_
    r2 = lr.score(X, y, sample_weight=weights)
    if return_model:
        return lr
    else:
        return y_pred, coef, r2

# ---------------------------------------------------------------------------- #
#                        Run data preparation pipelines                        #
# ---------------------------------------------------------------------------- #


# prepare_household_survey(country='Saint Lucia')