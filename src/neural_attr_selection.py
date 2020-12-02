import pandas as pd
import os
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, accuracy_score, \
    average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from pathlib import Path

from data import get_data_dir_path
from test_set_constructor import __move_instances
from weka import Weka
from shutil import copyfile

DATA_DIR = get_data_dir_path()
WEKA_PATH = Path(f'C:/Program Files/Weka-3-8-4/weka.jar')
TOP_SELECTED_FEATURES = [5, 10, 20, 50, 200, 500]
DIRECTORIES = [Path(f'{DATA_DIR}/neural'), Path(f'{DATA_DIR}/neural/arff'), Path(f'{DATA_DIR}/neural/csv'),
               Path(f'{DATA_DIR}/neural/arff/x'), Path(f'{DATA_DIR}/neural/arff/y'), Path(f'{DATA_DIR}/neural/csv/x'),
               Path(f'{DATA_DIR}/neural/csv/y')]


def __create_dir(path: Path) -> Path:
    """
    Creates the directory based on the path.
    
    :param path: Path for directory to be created if doesnt exist.
    :return: Path of the new directory.
    """

    if path.exists():
        return path

    os.mkdir(path.absolute())
    return path


def main():
    # Create weka object
    weka = Weka(WEKA_PATH)

    # Create directories if they dont exist 
    for directory in DIRECTORIES:
        __create_dir(directory)

    # Run for all top features
    for i in TOP_SELECTED_FEATURES:
        # Main training and test data set
        training_input_path = Path(f'{DATA_DIR}/train_gr_smpl.arff')
        test_input_path = Path(f'{DATA_DIR}/test_gr_smpl.arff')
        # Top X features from data set
        arff_output_path = Path(f'{DATA_DIR}/neural/arff/train_gr_smpl_top_{i}.arff')
        csv_output_path = Path(f'{DATA_DIR}/neural/csv/train_gr_smpl_top_{i}.csv')

        # Run attribute selection
        attributes = weka.select_top_correlating_attrs(training_input_path, arff_output_path, i)

        # Export attribute selection as csv file for later use
        weka.arff_to_csv(arff_output_path, csv_output_path)

        # Genreate list of indicies from 0 to number of features. This is used when seperating the arff file into X
        # and Y
        indices = list(range(0, i))

        # Export features alone without class
        weka.filter_attributes(arff_output_path, f'{DATA_DIR}/neural/arff/x/x_train_gr_smpl_top_{i}.arff', indices)
        weka.arff_to_csv(f'{DATA_DIR}/neural/arff/x/x_train_gr_smpl_top_{i}.arff',
                         Path(f'{DATA_DIR}/neural/csv/x/x_train_gr_smpl_top_{i}.csv'))

        # Export class simply incase input data is non-standard y data points (e.g. has been randomised)
        weka.filter_attributes(arff_output_path, f'{DATA_DIR}/neural/arff/y/y_train_smpl_top_{i}.arff', [i])
        weka.arff_to_csv(f'{DATA_DIR}/neural/arff/y/y_train_smpl_top_{i}.arff',
                         Path(f'{DATA_DIR}/neural/csv/y/y_train_smpl_top_{i}.csv'))

        # Grab the attributes from the training file, select those same attributes from the test files IN ORDER
        weka.filter_attributes(test_input_path, f'{DATA_DIR}/neural/arff/x/x_test_gr_smpl_top_{i}.arff', attributes)
        weka.arff_to_csv(f'{DATA_DIR}/neural/arff/x/x_test_gr_smpl_top_{i}.arff',
                         Path(f'{DATA_DIR}/neural/csv/x/x_test_gr_smpl_top_{i}.csv'))

        # Export class simply incase input data is non-standard y data points (e.g. has been randomised)
        weka.filter_attributes(test_input_path, f'{DATA_DIR}/neural/arff/y/y_test_smpl_top_{i}.arff', [2304])
        weka.arff_to_csv(f'{DATA_DIR}/neural/arff/y/y_test_smpl_top_{i}.arff',
                         Path(f'{DATA_DIR}/neural/csv/y/y_test_smpl_top_{i}.csv'))

    # Loop thorugh both 4000 and 9000 moved instanced 
    for task in [4000, 9000]:
        # Copy files to keep data separate 
        copyfile(f'{DATA_DIR}/x_train_gr_smpl.csv', f'{DATA_DIR}/x_task{task}_train_gr_smpl.csv')
        copyfile(f'{DATA_DIR}/x_test_gr_smpl.csv', f'{DATA_DIR}/x_task{task}_test_gr_smpl.csv')

        #Move instances from full training set to full test set
        __move_instances(f'{DATA_DIR}/x_task{task}_train_gr_smpl.csv', f'{DATA_DIR}/x_task{task}_test_gr_smpl.csv',
                         task)

        # Copy files to keep data separate 
        copyfile(f'{DATA_DIR}/y_train_smpl.csv', f'{DATA_DIR}/y_task{task}_train_smpl.csv')
        copyfile(f'{DATA_DIR}/y_test_smpl.csv', f'{DATA_DIR}/y_task{task}_test_smpl.csv')

        #Move instances from full training set to full test set
        __move_instances(f'{DATA_DIR}/y_task{task}_train_smpl.csv', f'{DATA_DIR}/y_task{task}_test_smpl.csv', task)

        for i in TOP_SELECTED_FEATURES:
            # Copy files to keep data separate 
            copyfile(f'{DATA_DIR}/neural/csv/x/x_train_gr_smpl_top_{i}.csv',
                     f'{DATA_DIR}/neural/csv/x/x_task{task}_train_gr_smpl_top_{i}.csv')
            copyfile(f'{DATA_DIR}/neural/csv/x/x_test_gr_smpl_top_{i}.csv',
                     f'{DATA_DIR}/neural/csv/x/x_task{task}_test_gr_smpl_top_{i}.csv')

            #Move instances from partial attribute training set to partial attribute test set
            __move_instances(f'{DATA_DIR}/neural/csv/x/x_task{task}_train_gr_smpl_top_{i}.csv',
                             f'{DATA_DIR}/neural/csv/x/x_task{task}_test_gr_smpl_top_{i}.csv', task)

            # Copy files to keep data separate 
            copyfile(f'{DATA_DIR}/neural/csv/y/y_train_smpl_top_{i}.csv',
                     f'{DATA_DIR}/neural/csv/y/y_task{task}_train_smpl_top_{i}.csv')
            copyfile(f'{DATA_DIR}/neural/csv/y/y_test_smpl_top_{i}.csv',
                     f'{DATA_DIR}/neural/csv/y/y_task{task}_test_smpl_top_{i}.csv')

            #Move instances from partial attribute training set to partial attribute test set
            __move_instances(f'{DATA_DIR}/neural/csv/y/y_task{task}_train_smpl_top_{i}.csv',
                             f'{DATA_DIR}/neural/csv/y/y_task{task}_test_smpl_top_{i}.csv', task)


if __name__ == "__main__":
    main()
