import os
import pandas as pd
import numpy as np

from scint_flux import constants
from scint_UM100.data_retreval import retrieve_data_funs


def calculate_explicit_flux(model, target_hour, target_DOY, path,
                            main_dir='D:/Documents/scint_UM100/scint_UM100/data_retreval/stash_data/'):
    # read csv
    # form csv filename
    filename = 'stash_data_' + model + '_' + str(target_hour).zfill(2) + '.csv'
    filepath = main_dir + filename

    assert os.path.isfile(filepath)

    # read in the df
    existing_df = pd.read_csv(filepath)
    existing_df.index = existing_df.grid
    existing_df = existing_df.drop(columns=['grid'])

    # find the effective measurement height of the observation for this hour
    # z_f csvs are screated within scint_fp package
    z_f = retrieve_data_funs.grab_obs_z_f_vals(target_DOY, target_hour, path)

    # calculate explicit flux
    rho = (existing_df['m01s00i253'] / (constants.radius_of_earth + z_f) ** 2).mean()

    T_prime = existing_df.air_temperature.mean() - existing_df.air_temperature
    W_prime = existing_df.upward_air_velocity.mean() - existing_df.upward_air_velocity

    T_prime_W_prime_av = (T_prime * W_prime).mean()
    explicit = T_prime_W_prime_av * rho * constants.cp

    existing_df['total_flux'] = existing_df['upward_heat_flux_in_air'] + explicit

    # rewrite the csv with this column
    existing_df.to_csv(filepath)
