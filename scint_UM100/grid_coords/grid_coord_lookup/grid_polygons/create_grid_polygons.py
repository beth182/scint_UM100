import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import Point
import os
import geopandas as gpd
from shapely.geometry import Polygon

import warnings
warnings.filterwarnings("ignore")



# model = '100m'
# model = '300m'
model = 'ukv'

# requires the output from running get_example_points.py to work
# ToDo: update this

save_path = os.getcwd().replace('\\', '/') + '/'
csv_location = save_path + '../' + 'rotation_tests_BTT_' + model + '.csv'

# actual run
if model == '100m':
    step_number = 80
elif model == '300m':
    step_number = 25
elif model == 'ukv':
    step_number = 6
else:
    print('end')

# temp reduced
# step_number = 5

df = pd.read_csv(csv_location)

# left to right plane
right_y_diff = df.y[0] - df.y[6]
right_x_diff = df.x[6] - df.x[0]

up_x_diff = df.x[8] - df.x[0]
up_y_diff = df.y[8] - df.y[0]

# empty lists to append and make df from
lon_x_list = [df.x[0]]
lon_y_list = [df.y[0]]

# to the right # steps
for i in range(1, step_number + 1):
    new_x = df.x[0] + (i * right_x_diff)
    new_y = df.y[0] - (i * right_y_diff)

    lon_x_list.append(new_x)
    lon_y_list.append(new_y)

# to the left # steps
for i in range(1, step_number + 1):
    new_x = df.x[0] - (i * right_x_diff)
    new_y = df.y[0] + (i * right_y_diff)

    lon_x_list.append(new_x)
    lon_y_list.append(new_y)

# left to right df
lon_df = pd.DataFrame.from_dict({'x': lon_x_list, 'y': lon_y_list})

# sort df
lon_df = lon_df.sort_values('x').reset_index().drop(columns=['index'])

lon_df['row #'] = np.ones(len(lon_df)) * step_number
lon_df['col #'] = lon_df.index

row_df_list = []
for index, row in lon_df.iterrows():

    row_x = row.x
    row_y = row.y
    col_num = row['col #']

    row_x_list = []
    row_y_list = []
    col_num_list = []
    row_num_list = []

    # go up from row # steps
    for i in range(1, step_number + 1):
        col_num_list.append(col_num)

        new_x = row_x + (i * up_x_diff)
        new_y = row_y + (i * up_y_diff)

        row_num_list.append(step_number + i)
        row_x_list.append(new_x)
        row_y_list.append(new_y)

    # down # steps
    for i in range(1, step_number + 1):
        col_num_list.append(col_num)

        new_x = row_x - (i * up_x_diff)
        new_y = row_y - (i * up_y_diff)

        row_num_list.append(step_number - i)

        row_x_list.append(new_x)
        row_y_list.append(new_y)

    row_df = pd.DataFrame.from_dict({'x': row_x_list, 'y': row_y_list, 'row #': row_num_list, 'col #': col_num_list})
    row_df_list.append(row_df)

all_row_df = pd.concat(row_df_list)
all_df = pd.concat([lon_df, all_row_df])

all_df = all_df.sort_values(['row #', 'col #']).reset_index().drop(columns=['index'])

# fig, ax = plt.subplots()
# ax.scatter(all_df.x, all_df.y, c='k')

def midpoint(p1, p2):
    return Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)

# isolate squares
count = 1
for j in range(0, (step_number * 2)):
    # loop over rows
    for i in range(0, (step_number * 2)):

        if len(all_df[all_df['row #'] == i + 1][all_df['col #'] == j + 1]) == 0:
            continue

        elif len(all_df[all_df['row #'] == i - 1][all_df['col #'] == j - 1]) == 0:
            continue

        else:

            # Square 1
            # to calculate BL
            square_1_df = pd.concat([all_df[all_df['row #'] == i - 1][all_df['col #'] == j - 1],  # BL
                                     all_df[all_df['row #'] == i][all_df['col #'] == j - 1],  # TL
                                     all_df[all_df['row #'] == i][all_df['col #'] == j],  # TR
                                     all_df[all_df['row #'] == i - 1][all_df['col #'] == j]])  # BR

            # calculate mid point of square 1
            BL = midpoint(square_1_df.iloc[0], square_1_df.iloc[2])

            # Square 2
            # to calculate TL
            square_2_df = pd.concat([all_df[all_df['row #'] == i][all_df['col #'] == j - 1],  # BL
                                     all_df[all_df['row #'] == i + 1][all_df['col #'] == j - 1],  # TL
                                     all_df[all_df['row #'] == i + 1][all_df['col #'] == j],  # TR
                                     all_df[all_df['row #'] == i][all_df['col #'] == j]])  # BR

            # calculate mid point of square 2
            TL = midpoint(square_2_df.iloc[0], square_2_df.iloc[2])

            # Square 3
            # to calculate TR
            square_3_df = pd.concat([all_df[all_df['row #'] == i][all_df['col #'] == j],  # BL
                                     all_df[all_df['row #'] == i + 1][all_df['col #'] == j],  # TL
                                     all_df[all_df['row #'] == i + 1][all_df['col #'] == j + 1],  # TR
                                     all_df[all_df['row #'] == i][all_df['col #'] == j + 1]])  # BR

            # calculate mid point of square 3
            TR = midpoint(square_3_df.iloc[0], square_3_df.iloc[2])

            # Square 4
            # to calculate BR
            square_4_df = pd.concat([all_df[all_df['row #'] == i - 1][all_df['col #'] == j],  # BL
                                     all_df[all_df['row #'] == i][all_df['col #'] == j],  # TL
                                     all_df[all_df['row #'] == i][all_df['col #'] == j + 1],  # TR
                                     all_df[all_df['row #'] == i - 1][all_df['col #'] == j + 1]])  # BR

            # calculate mid point of square 4
            BR = midpoint(square_4_df.iloc[0], square_4_df.iloc[2])


            # combine into a df
            x_list = [BL.x, TL.x, TR.x, BR.x]
            y_list = [BL.y, TL.y, TR.y, BR.y]

            square_df = pd.DataFrame({'x': x_list, 'y': y_list,
                                      'descrip': ['BL', 'TL', 'TR', 'BR'],
                                      'grid': list(np.ones(4) * count)})

            polygon_geom = Polygon(zip(square_df.x, square_df.y))
            polygon = gpd.GeoDataFrame(index=[0], crs='epsg:32631', geometry=[polygon_geom])
            polygon.to_file(filename=save_path + 'UM' + model.split('m')[0] + '_shapes/' + str(count) + ".gpkg", driver="GPKG")

        print(count)
        count += 1



print('end')
