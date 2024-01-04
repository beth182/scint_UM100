import numpy as np
import datetime as dt
import os
import pandas as pd

from model_eval_tools.retrieve_UKV import read_premade_model_files
from scint_flux import look_up


def handle_time(nc_file):
    """
    Function to handle model time and convert to a list of datetimes
    :return:
    """
    # reads in time
    # get time units for time conversion and start time
    unit_start_time = nc_file.variables['time'].units

    # Read in minutes since the start time and add it on
    # Note: time_to_datetime needs time_since to be a list. Hence put value inside a single element list first
    time_since_start = [np.squeeze(nc_file.variables['forecast_reference_time'])]

    run_start_time = read_premade_model_files.time_to_datetime(unit_start_time, time_since_start)[0]

    # get number of forecast hours to add onto time_start
    run_len_hours = np.squeeze(nc_file.variables['forecast_period'][:]).tolist()

    if type(run_len_hours) == float:
        run_len_hours = [run_len_hours]

    run_times = [run_start_time + dt.timedelta(seconds=hr * 3600) for hr in run_len_hours]

    return run_times


def merge(list1, list2):
    """
    makes tuples from 2 lists (of coords)
    :param list1:
    :param list2:
    :return:
    """
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list


def grab_obs_z_f_vals(target_DOY, target_hour, path,
                      main_dir='D:/Documents/scint_UM100/scint_UM100/data_retreval/z_f_csvs/'):
    """

    :return:
    """

    path_num = dict((v, k) for k, v in look_up.scint_path_numbers.items())[path]

    file_name = 'z_f_' + str(path_num) + '_' + str(target_DOY) + '.csv'
    file_path = main_dir + file_name
    assert os.path.isfile(file_path)

    # pandas read file
    df = pd.read_csv(file_path)
    df.index = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    z_f = df.iloc[np.where(df.index.hour == target_hour)[0][0]].z_f

    return z_f
