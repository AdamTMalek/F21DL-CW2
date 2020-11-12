import os

from src.data import get_data_dir_path

DATA_DIR = get_data_dir_path()


def construct_test_set(number_of_instances: int):
    f"""
    Creates a new directory in the data directory and moves the specified number of instances
    from the original training set (x_train_gr_smpl.csv) the testing set (y_test_smpl.csv).
    
    The script does not modify the original files, instead it creates a copy of them.
    :param number_of_instances: Number of instances to move from the training set to the testing set
    """
    test_set_data_dir = f'{DATA_DIR}/{number_of_instances}_instances'
    os.mkdir(test_set_data_dir)

    with open(f'{DATA_DIR}/y_test_smpl.csv', 'r') as original_test_set_file:
        with open(f'{test_set_data_dir}/y_test_smpl.csv', 'w+') as new_test_set_file:
            # First, copy each line of the original test set file to the new one
            for line in original_test_set_file:
                new_test_set_file.write(line)

            with open(f'{DATA_DIR}/x_train_gr_smpl.csv', 'r') as original_training_set_file:
                with open(f'{test_set_data_dir}/x_train_gr_smpl.csv', 'w+') as new_training_set_file:
                    # Write the header file to the new training set file
                    new_training_set_file.write(original_training_set_file.readline())

                    # Append the required number of instances to the new test set file
                    for i in range(number_of_instances):
                        new_test_set_file.write(original_training_set_file.readline())

                    # Copy the rest into the new training set file
                    for line in original_training_set_file:
                        new_training_set_file.write(line)
