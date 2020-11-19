import os
import random
import shutil
from pathlib import Path
from typing import Iterable

from src.data import get_data_dir_path
from src.weka import Weka

DATA_DIR = get_data_dir_path()
DATA_SIZE = 9690
TOP_ATTRS_TO_TAKE = 50  # Number of top correlating attributes (pixels) to take


def construct_test_set(number_of_instances: int, weka_jar_path: Path):
    """
    Creates a new directory in the data directory and moves the specified number of instances
    from the original training set (x_train_gr_smpl.csv) the testing set (y_test_smpl.csv).
    
    The script does not modify the original files, instead it creates a copy of them.
    :param number_of_instances: Number of instances to move from the training set to the testing set
    """
    test_dir_path = __create_test_dir(number_of_instances)

    merged_train_csv_file = test_dir_path.joinpath('train.csv')
    merged_test_csv_file = test_dir_path.joinpath('test.csv')
    merge_class_attribute(f'{DATA_DIR}/x_train_gr_smpl.csv', f'{DATA_DIR}/y_train_smpl.csv', str(merged_train_csv_file))
    merge_class_attribute(f'{DATA_DIR}/x_test_gr_smpl.csv', f'{DATA_DIR}/y_test_smpl.csv', str(merged_test_csv_file))
    __move_instances(str(merged_train_csv_file), str(merged_test_csv_file), number_of_instances)

    weka = Weka(weka_jar_path)
    __apply_numeric_to_nominal_filter((merged_train_csv_file, merged_test_csv_file), weka)

    # Apply the attribute selection to get the top correlating attributes (pixels)
    arff_train_file = test_dir_path.joinpath('train.arff')  # Created by the __apply_numeric_to_nominal_filter function
    arff_train_reduced_attrs_file = test_dir_path.joinpath(f'train_{TOP_ATTRS_TO_TAKE}_attrs.arff')
    attributes = weka.select_top_correlating_attrs(arff_train_file, arff_train_reduced_attrs_file, TOP_ATTRS_TO_TAKE)

    # Take the same attributes and filter them from the test
    arff_test_file = test_dir_path.joinpath('test.arff')  # Created by the __apply_numeric_to_nominal_filter function
    arff_test_reduced_attrs_file = test_dir_path.joinpath(f'test_{TOP_ATTRS_TO_TAKE}_attrs.arff')
    weka.filter_attributes(arff_test_file, arff_test_reduced_attrs_file, attributes)


def __apply_numeric_to_nominal_filter(files: Iterable[Path], weka: Weka):
    for file in files:
        directory = file.parent
        arff_name = file.stem + '.arff'
        arff_file_path = directory.joinpath(arff_name)

        weka.set_class_attr_to_nominal(file, arff_file_path)


def merge_class_attribute(data_file_path: str, attributes_file_path: str, dest_path: str):
    """
    Merges the class attribute together with the data file.

    :param data_file_path: CSV file with picture data
    :param attributes_file_path: Attributes file (one with attribute ranging from 0-9)
    :param dest_path: The destination file path
    """
    data_file = open(data_file_path, 'r')
    attributes_file = open(attributes_file_path, 'r')

    attributes_file.readline()  # Ignore the first line which contains no useful information

    with open(dest_path, 'w+') as dest:
        dest.write(f'{data_file.readline().rstrip()},class\n')  # Read the header line, add the class attribute

        # Copy all lines adding the class attribute
        for line in data_file:
            dest.write(f'{line.rstrip()},{attributes_file.readline().rstrip()}\n')

    data_file.close()
    attributes_file.close()


def __create_test_dir(number_of_instances: int) -> Path:
    """
    Creates the test directory with the name based on the number of instances.

    The name will be in the format {number_of_instances}_instances.
    The new directory will be placed inside the data directory.

    :param number_of_instances: Number of instances that will be moved from data to test set.
    :return: Path of the new directory.
    """
    path = Path(f'{DATA_DIR}/{number_of_instances}_instances')

    if path.exists():
        return path

    os.mkdir(path.absolute())
    return path


def __move_instances(data_file_path: str, test_file_path: str, number_of_instances: int):
    """
    Moves the specified number of instances from one file to the other.

    :param data_file_path: File from which the first X instances will be taken.
    :param test_file_path: File to which the instances will be appended.
    :param number_of_instances: Number of instances to take
    """
    # Create a copy of the data file
    data_file_copy_path = f'{data_file_path}.copy'
    shutil.move(data_file_path, data_file_copy_path)

    lines_to_take = set(random.sample(range(DATA_SIZE), number_of_instances))  # Lines to take to the test file

    with open(data_file_copy_path, 'r') as data_file_copy:
        with open(data_file_path, 'w+') as data_file:
            with open(test_file_path, 'a') as test_file:
                data_file.write(data_file_copy.readline())  # Copy the header line

                for index, line in enumerate(data_file_copy):
                    if index in lines_to_take:
                        test_file.write(line)
                    else:
                        data_file.write(line)

    # Delete the copy
    os.remove(data_file_copy_path)
