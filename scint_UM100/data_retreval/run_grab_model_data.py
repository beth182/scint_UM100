import os
import pandas as pd

from scint_UM100.data_retreval import retreve_data_from_target_grids


def run_all_grab(path, target_DOY, target_hours, model):
    """

    :param path:
    :param target_DOY:
    :param target_hours:
    :param model:
    :return:
    """

    for target_hour in target_hours:

        # only run for hours that need it
        csv_name = path + '_' + str(target_DOY)[-3:].zfill(3) + '_UM100_QH_' + model + '.csv'
        csv_path = os.getcwd().replace('\\', '/') + '/' + csv_name

        # check to see if the index exists
        if os.path.isfile(csv_path):
            existing_df = pd.read_csv(csv_path)
            existing_df.index = existing_df.hour
            existing_df = existing_df.drop(columns=['hour'])
            if target_hour in existing_df.index:
                print('Already done ', str(target_hour))
                print(' ')

                continue
            else:
                pass
        else:
            pass

        # if hour doesnt already exist:
        retreve_data_from_target_grids.grab_model_data(path, target_DOY, target_hour, model)

    print('end')


if __name__ == '__main__':
    # user inputs
    path = 'BCT_IMU'
    target_DOY = 2016134

    target_hours = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    # target_hours = [12]

    # model = '100m'
    # model = '300m'
    model = 'ukv'

    run_all_grab(path, target_DOY, target_hours, model)

    print('end')
