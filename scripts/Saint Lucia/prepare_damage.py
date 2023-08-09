import pandas as pd
import numpy as np
import random

def prepare_asset_damage(country: str, scale: str, return_period: int = 100) -> None:
    '''Prepare district-level asset damage data and save it into a XLSX file.'''
    if country == 'Saint Lucia':
        if scale == 'district':
            # Load raw data
            df = pd.read_excel(
                '../../data/raw/asset_damage/Saint Lucia/St Lucia 2015 exposure summary.xlsx', sheet_name='total by parish', skiprows=1)
            # Remove redundant columns
            df.drop(df.columns[0], axis=1, inplace=True)
            # Even though the data is by `parish``, let's call the corresponding column `district``
            df.rename(columns={'Unnamed: 1': 'district'}, inplace=True)
            # !: Check whether rp is = 100 given the data
            df['rp'] = 100
            df.rename(
                columns={'Combined Total': 'exposed_value'}, inplace=True)

            # !: Replace with the real data
            # Let's assume that PML is equal to AAL % by district * by the PML for the whole country
            # These values are from PML Results 19022016 SaintLucia FinalSummary2.xlsx
            total_pml = {10: 351733.75,  # 3,517,337.50
                         50: 23523224.51,  # 2,352,322,451.00
                         100: 59802419.04,  # 5,980,241,904.00
                         250: 147799213.30,  # 14,779,921,330.00
                         500: 248310895.20,  # 24,831,089,520.00
                         1000: 377593847.00}  # 37,759,384,700.00
            aal = pd.read_excel(
                '../../data/processed/asset_damage/Saint Lucia/AAL Results 19022016 StLucia FinalSummary2 adjusted.xlsx', sheet_name='AAL St. Lucia Province')
            aal.set_index('Name', inplace=True)
            aal = aal[['AAL as % of Total AAL']]
            aal.columns = ['pml']
            aal = aal[aal.index.notnull()]
            pml = aal.multiply(total_pml[return_period])
            df = pd.merge(df, pml, left_on='district', right_index=True)
            df.to_excel(
                f'../../data/processed/asset_damage/{country}/{country}.xlsx', index=False)
        else:
            raise ValueError(
                'Only `district` scale is supported for Saint Lucia.')
    else:
        raise ValueError('Only `Saint Lucia` is supported.')
    
# prepare_asset_damage(country='Saint Lucia',
#                      scale='district', return_period=100)
