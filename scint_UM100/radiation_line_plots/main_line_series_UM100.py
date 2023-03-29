# a copy from scint plots, with features updates as the UM100 analysis needs

import datetime as dt

from scint_flux.functions import read_calculated_fluxes
from scint_flux import look_up

from scint_plots.tools.preprocessed_UKV_csvs import read_UKV_csvs
from scint_UM100.radiation_line_plots import line_time_series_funs_UM100

scint_path = 12
DOY_list = [2016134]
var_list = ['QH', 'kdown']
time_res = '1min_sa10_mins_ending'

pair_id = look_up.scint_path_numbers[scint_path]

DOY_dict = {}

for DOY in DOY_list:


    # get model QH and kdown
    df_UKV_all = read_UKV_csvs.read_all_of_preprocessed_UKV_csv(['BL_H', 'kdown'])[['BL_H_' + str(scint_path), 'kdown_' + str(scint_path)]].rename(columns={'BL_H_' + str(scint_path): 'BL_H_UKV', 'kdown_' + str(scint_path): 'kdown_UKV'})

    # constrict for just the target DOY
    df_UKV = df_UKV_all.loc[dt.datetime.strptime(str(DOY), '%Y%j'): dt.datetime.strptime(str(DOY), '%Y%j') + dt.timedelta(days=1)]


    # read the observations
    df = read_calculated_fluxes.extract_data(doy_list=[DOY],
                                             pair_id=pair_id,
                                             var_list=var_list,
                                             time_res=time_res)


    line_time_series_funs_UM100.times_series_line_QH_KDOWN(df, pair_id, model_df=df_UKV)

print('end')
