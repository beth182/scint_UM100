import os
import pandas as pd
import numpy as np
import datetime as dt
import seaborn
import matplotlib.pyplot as plt


def get_hour_weighted_LC(year, DOY, path, model, hour_choice):
    dt_obj = dt.datetime.strptime(str(year) + str(DOY) + str(hour_choice).zfill(2), '%Y%j%H')

    current_path = os.getcwd().replace('\\', '/') + '/'

    # read the csv file containing the grids in SA at each time
    SA_csv_location = current_path + '../grid_coords/SA_grid_overlap/' + path + '_SA_UM' + model.split('m')[
        0] + '_grid_percentages.csv'
    assert os.path.isfile(SA_csv_location)

    # get df of SA weights for each grid
    existing_df = pd.read_csv(SA_csv_location)
    existing_df.index = existing_df['Unnamed: 0']
    existing_df = existing_df.drop(columns=['Unnamed: 0'])
    existing_df.index.name = 'grid'

    # isolate target time
    try:
        all_hour_df = existing_df[str(hour_choice).zfill(2)]
    except KeyError:
        all_hour_df = existing_df[str(hour_choice)]

    hour_df = all_hour_df.iloc[np.where(all_hour_df > 0)[0]]

    # make sure all the weights are close to 100
    assert np.isclose(hour_df.sum(), 100, 0.0001)

    # read lc csv
    LC_csv_location = current_path + model + '_all_grids_lc.csv'
    assert os.path.isfile(LC_csv_location)

    # get df with lc fractions for all grids
    lc_df = pd.read_csv(LC_csv_location)
    lc_df = lc_df.replace('--', np.nan, regex=True)
    lc_df = lc_df.astype(float)
    lc_df.index = lc_df['Unnamed: 0']
    lc_df = lc_df.drop(columns=['Unnamed: 0'])
    lc_df.index.name = 'grid'

    lc_df.index = lc_df.index.astype(int)

    # rows needing to be dropped:
    nan_rows = lc_df[lc_df.isna().any(axis=1)]

    if len(nan_rows) != 0:
        # check if any of the problem grids are in the hour df
        assert sum(np.in1d(nan_rows.index, hour_df.index)) == 0

        lc_df = lc_df.dropna()

    assert sum(np.isclose(lc_df.sum(axis=1), 1)) == len(lc_df)

    # isolate grids just in this hour's SA

    temp = pd.concat([lc_df, hour_df], axis=1).dropna()

    try:
        lc_hour = temp.drop(columns=[str(hour_choice)])
    except KeyError:
        lc_hour = temp.drop(columns=[str(hour_choice).zfill(2)])

    # check that the indexes for both df are all the same
    assert len(lc_hour) == len(hour_df)
    assert sum(lc_hour.index == hour_df.index) == len(lc_hour)

    weight = hour_df / 100

    weighted_df = lc_hour.mul(weight, axis=0)

    assert np.isclose(weighted_df.sum().sum(), 1, 0.0001)

    # weighted values across all grids
    all_weighted_lc = weighted_df.sum()

    # transpose so colums are LC types
    df_row = all_weighted_lc.to_frame().T
    row_ind = dt_obj.strftime('%y%m%d%H')
    df_row.index = [row_ind]

    return {'all': lc_hour, 'weighted': df_row, 'hour': row_ind}


if __name__ == '__main__':
    # user choices
    # ToDo make this flexible for other days to be possible
    year = 2016
    DOY = 134

    path = 'BCT_IMU'
    model = '100m'

    list_of_hours = np.arange(6, 19, 1).tolist()

    df_list_weighted = []

    all_hours = {}

    for hour in list_of_hours:
        return_dict = get_hour_weighted_LC(year, DOY, path, model, hour)
        df_list_weighted.append(return_dict['weighted'])

        all_hours[return_dict['hour']] = return_dict['all']

    full_df_weighted = pd.concat(df_list_weighted)

    # see which types have a non zero sum across all source areas
    non_zero_types = []
    for key in all_hours.keys():

        lc_types = all_hours[key].sum().index

        for lc_type in lc_types:

            if not np.isclose(all_hours[key].sum()[lc_type], 0):
                non_zero_types.append(lc_type)

    non_zero_types = list(set(non_zero_types))

    # get the dataframe into the format needed for the boxplot
    # get a boxplot of each hour

    hour_dfs = []

    for key in all_hours.keys():
        df_select = all_hours[key][non_zero_types]

        hour_col_list = [key] * len(df_select)

        df_select['hour'] = hour_col_list

        hour_dfs.append(df_select)

    test = pd.concat(hour_dfs)

    # seperate out columns

    df_types = []

    for type in non_zero_types:
        df_type = pd.DataFrame.from_dict({'vals': test[type], 'type': len(test[type]) * [type], 'hour': test['hour']})

        df_types.append(df_type)

    ye = pd.concat(df_types)

    ye = ye.sort_values('type')

    print('end')

    my_pal = {"lake": "blue", "canyon": "grey", "roof": "black", "soil": "brown", "needleleaf": "olive",
              "broadleaf": "forestgreen", 'C3': 'orange'}

    fig, ax = plt.subplots(figsize=(9, 10))

    # Create the boxplot using Seaborn
    seaborn.boxplot(
        x="hour",
        y="vals",
        hue="type",
        data=ye,
        width=1,
        whis=[0, 100],
        palette=my_pal,
        ax=ax)

    box_container = ax.patches if len(ax.patches) > 0 else ax.artists

    lines_per_boxplot = len(ax.lines) // len(box_container)


    # Yield successive n-sized
    # chunks from l.
    def divide_chunks(l, n):
        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]


    # list should have
    n = 7
    x = list(divide_chunks(list(np.arange(0, 98)), n))


    collections = []

    for i in range(0, len(x[0])):
        collection = []
        for list in x:
            collection.append(list[i])
        collections.append(collection)






    for i, box in enumerate(box_container):


        assert len([element for element in collections if i in element]) == 1
        group = [element for element in collections if i in element][0][0]
        type = sorted(non_zero_types)[group]
        col = my_pal[type]

        # number of hours = len(set(ye.hour))

        # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same colour as above
        for j in range(group * len(set(ye.hour)) * 6, group * len(set(ye.hour)) * 6 + len(set(ye.hour)) * 6):
            line = ax.lines[j]
            line.set_color(col)
            line.set_mfc(col)
            line.set_mec(col)




    # plt.show()
    print('end')
