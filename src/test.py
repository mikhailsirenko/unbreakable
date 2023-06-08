import pandas as pd


class Tester():
    def _test_availability_of_geographical_unit(self, country: str = '', state: str = '', district: str = '', scale: str = ''):
        if scale == 'country':
            f = country
        elif scale == 'state':
            f = state
        elif scale == 'district':
            f = district
        else:
            raise ValueError(
                f'Scale {scale} is not supported. Please use country, district or state.')

        if scale != 'country':
            available_geographical_units = pd.read_csv(
                f'../data/processed/household_survey/{country}/{country}.csv')[scale].tolist()
        else:
            available_geographical_units = ''

        if scale == 'state':
            if state not in available_geographical_units:
                raise ValueError(
                    f'Region {state} is not supported. Please use one of {available_geographical_units}.')

        elif scale == 'district':
            if district not in available_geographical_units:
                raise ValueError(
                    f'District {district} is not supported. Please use one of {available_geographical_units}.')
