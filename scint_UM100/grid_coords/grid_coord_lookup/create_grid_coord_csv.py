# script similar to create_grid_polygons - but instead of saving all grids as files, I save a csv with coords
# to use as lookup for getting data out

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from shapely.geometry import Point

import warnings
warnings.filterwarnings("ignore")

# requires the output from running get_example_points.py to work
# ToDo: update this

save_path = os.getcwd().replace('\\', '/') + '/'
csv_location = save_path + 'rotation_tests_BTT.csv'

# actual run
step_number = 80

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
square_df_list = []
count = 1
for j in range(0, (step_number * 2)):
    # loop over rows
    for i in range(0, (step_number * 2)):
        # # bottom left
        # ax.scatter(all_df[all_df['row #'] == i][all_df['col #'] == j].x,
        #            all_df[all_df['row #'] == i][all_df['col #'] == j].y, c='r')
        # # top left
        # ax.scatter(all_df[all_df['row #'] == i + 1][all_df['col #'] == j].x,
        #            all_df[all_df['row #'] == i + 1][all_df['col #'] == j].y, c='r')
        # # top right
        # ax.scatter(all_df[all_df['row #'] == i + 1][all_df['col #'] == j + 1].x,
        #            all_df[all_df['row #'] == i + 1][all_df['col #'] == j + 1].y, c='r')
        # # bottom right
        # ax.scatter(all_df[all_df['row #'] == i][all_df['col #'] == j + 1].x,
        #            all_df[all_df['row #'] == i][all_df['col #'] == j + 1].y, c='r')

        square_df = pd.concat([all_df[all_df['row #'] == i][all_df['col #'] == j],
                               all_df[all_df['row #'] == i + 1][all_df['col #'] == j],
                               all_df[all_df['row #'] == i + 1][all_df['col #'] == j + 1],
                               all_df[all_df['row #'] == i][all_df['col #'] == j + 1]])

        # square_df = square_df.rename_axis(index='grid')
        square_df['descrip'] = ['BL', 'TL', 'TR', 'BR']

        square_df['grid'] = (np.ones(len(square_df)) * count).astype(int)

        square_df = square_df.drop(columns=['row #', 'col #'])

        # calculate mid point of grid
        # mid point of bottom left and top right
        # bottom left
        b_l_point = square_df.iloc[0]
        t_r_point = square_df.iloc[2]

        mid_point_1 = midpoint(b_l_point, t_r_point)

        # mid point of bottom right and top left
        b_r_point = square_df.iloc[3]
        t_l_point = square_df.iloc[1]

        mid_point_2 = midpoint(b_r_point, t_l_point)

        # make sure they're the same
        assert mid_point_1 == mid_point_2

        # mid point df
        mid_point_df = pd.DataFrame({'x': mid_point_1.x, 'y': mid_point_1.y,
                                               'descrip': 'MID', 'grid': int(count)}, index=[0])

        square_df_all = pd.concat([square_df, mid_point_df])

        square_df_all = square_df_all.reset_index().drop(columns=['index'])

        square_df_list.append(square_df_all)

        # plot
        # plt.scatter(mid_point_1.x, mid_point_1.y, c='green')

        print(count)
        count += 1


# combine all
df_all_squares = pd.concat(square_df_list)



print('end')

df_all_squares.to_csv(save_path + 'grid_coords.csv')