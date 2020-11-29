from typing import Dict
import neural_attr_selection
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, \
    average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from data import get_data_dir_path, get_evidence_dir_path
from neural_attr_selection import TOP_SELECTED_FEATURES

DATA_PATH = get_data_dir_path()
EVIDENCE_PATH = get_evidence_dir_path()


def get_scores(classifier, ten_fold: bool, images: DataFrame, classes: DataFrame) -> Dict:
    result = {}
    if ten_fold:
        prediction = cross_val_predict(classifier, images, classes, cv=10, n_jobs=-1)
        probability = cross_val_predict(classifier, images, classes, cv=10, n_jobs=-1, method='predict_proba')
    else:
        prediction = classifier.predict(images)
        probability = classifier.predict_proba(images)
    # print(prediction) # For debugging
    result["accuracy"] = accuracy_score(classes, prediction)
    result["precision"] = precision_score(classes, prediction, average='weighted')
    result["f_score"] = f1_score(classes, prediction, average='weighted')
    result["recall"] = recall_score(classes, prediction, average='weighted')
    result["roc_area"] = roc_auc_score(y_true=classes, y_score=probability, average='weighted', multi_class='ovo')
    result["classification_report"] = classification_report(y_true=classes, y_pred=prediction)
    cm = confusion_matrix(y_true=classes, y_pred=prediction, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    tp_fp = get_tp_fp(cm)
    result["confusion_matrix"] = cm
    result["tpr"] = tp_fp["tp_rate"]
    result["fpr"] = tp_fp["fp_rate"]
    return result


def get_tp_fp(confusion_matrix):
    # From https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    # True positive rate
    true_positive_rate = TP / (TP + FN)
    # True negative rate
    false_positive_rate = FP / (FP + TN)

    results = {
        "tp_rate": true_positive_rate,
        "fp_rate": false_positive_rate
    }
    return results


def print_scores(scores: Dict, classifier_type: str, seed_value: int, model_parameters, mlp: bool) -> None:
    print(f'Accuracy: {scores["accuracy"]}')
    print(f'Precision: {scores["precision"]}')
    print(f'F score: {scores["f_score"]}')
    print(f'Recall: {scores["recall"]}')
    print(f'True Positive: {scores["tpr"]}')
    print(f'False Positive: {scores["fpr"]}')
    print(f'ROC area: {scores["roc_area"]}')
    print(scores["confusion_matrix"])
    print(scores["classification_report"])
    df = DataFrame([[classifier_type, seed_value, model_parameters, scores["accuracy"], scores["precision"],
                     scores["f_score"], scores["recall"], scores["roc_area"], scores["tpr"], scores["fpr"],
                     scores["confusion_matrix"], scores["classification_report"]]])
    if mlp:
        df.to_csv(f"{EVIDENCE_PATH}/mlp.csv", mode='a', header=False)
    else:
        df.to_csv(f"{EVIDENCE_PATH}/linear.csv", mode='a', header=False)


def evaluate_linear_classifier(ten_fold: bool, test_images: DataFrame, test_classes: DataFrame,
                               training_images: DataFrame, training_classes: DataFrame,
                               tolerance: float, c_value: float, seed_value: int, num_of_attrs: int):
    # Normalise training data set
    training_images = training_images / 255
    training_classes = training_classes.values.ravel()

    # Linear
    linear = LogisticRegression(max_iter=1000,
                                tol=tolerance,
                                C=c_value,
                                class_weight='balanced',
                                random_state=seed_value).fit(training_images, training_classes)

    # Normalise test data set
    test_images = test_images / 255
    test_classes = test_classes.values.ravel()

    # Get scores and results
    linear_result = get_scores(linear, ten_fold=ten_fold, images=test_images, classes=test_classes)
    print_scores(linear_result, "Linear", seed_value, [num_of_attrs, tolerance, c_value, ten_fold], False)


def evaluate_MLP_classifier(ten_fold: bool, test_images: DataFrame, test_classes: DataFrame, training_images: DataFrame,
                            training_classes: DataFrame, seed_value: int, tolerance: float,
                            layer_sizes, max_iterations: int, num_of_attrs: int):
    # Normalise training data set
    training_images = training_images / 255
    training_classes = training_classes.values.ravel()

    # MLP (THIS IS VEEEERY SLOW)
    mlp_classifier = MLPClassifier(solver='lbfgs',  # DO NOT REMOVE THIS LINE UNLESS YOU KNOW WHAT YOU ARE DOING
                                   activation='logistic',
                                   max_iter=max_iterations,
                                   hidden_layer_sizes=layer_sizes,
                                   tol=tolerance,
                                   batch_size=100,
                                   random_state=seed_value).fit(training_images, training_classes)

    # Normalise test data set
    test_images = test_images / 255
    test_classes = test_classes.values.ravel()

    # Get scores and results
    mlp_result = get_scores(mlp_classifier, ten_fold, images=test_images, classes=test_classes)
    print_scores(mlp_result, "MLP", seed_value, [num_of_attrs, tolerance, max_iterations, layer_sizes], True)


SEED_VALUES = [818, 702, 754]
TOLERANCE_VALUES = [0.001, 0.0001]
C_VALUES = [0.1, 0.2]
MAX_ITERATION_VALUES = [1000, 2000]
LAYER_SIZES_VALUES = [(50,), (100,), (100, 50), (200, 100, 50)]


def run_linear_task1(seed_value: int):
    for tolerance in TOLERANCE_VALUES:
        print(f"=================================== Iterating with tolerance of {tolerance}"
              f" set=================================")
        for c_value in C_VALUES:
            print(f"=================================== Iterating with c_value of {c_value} set "
                  f"===================================")
            # Fit and test using training set, without then with ten fold:
            for tenfold in [False, True]:
                images = pd.read_csv(f"{DATA_PATH}/x_train_gr_smpl.csv")
                classes = pd.read_csv(f"{DATA_PATH}/y_train_smpl.csv")
                test_images = images
                test_classes = classes
                print(f"=================================== Training Linear Classifier with All features,"
                      f" TenFold={tenfold} ===================================")
                evaluate_linear_classifier(ten_fold=tenfold, test_images=test_images, test_classes=test_classes,
                                           training_images=images, training_classes=classes, tolerance=tolerance,
                                           c_value=c_value, seed_value=seed_value, num_of_attrs=9690)

                # Import TOP SELECTED FEATURES from neural_attr_selection
                for i in TOP_SELECTED_FEATURES:
                    images = pd.read_csv(f"{DATA_PATH}/neural/csv/x/x_train_gr_smpl_top_{i}.csv")
                    classes = pd.read_csv(f"{DATA_PATH}/neural/csv/y/y_train_smpl_top_{i}.csv")
                    test_images = images
                    test_classes = classes
                    print(f"========== Training Linear Classifier with {i} features, TenFold={tenfold}"
                          f"===================================")
                    evaluate_linear_classifier(ten_fold=tenfold, test_images=test_images, test_classes=test_classes,
                                               training_images=images, training_classes=classes, tolerance=tolerance,
                                               c_value=c_value, seed_value=seed_value, num_of_attrs=i)


def run_linear_task3(seed_value: int):
    for tolerance in TOLERANCE_VALUES:
        print(f"=================================== Iterating with tolerance of {tolerance}"
              f" set=================================")
        for c_value in C_VALUES:
            print(f"=================================== Iterating with c_value of {c_value}"
                  f" set=================================")
            images = pd.read_csv(f"{DATA_PATH}/x_train_gr_smpl.csv")
            classes = pd.read_csv(f"{DATA_PATH}/y_train_smpl.csv")
            test_images = pd.read_csv(f'{DATA_PATH}/x_test_gr_smpl.csv')
            test_classes = pd.read_csv(f'{DATA_PATH}/y_test_smpl.csv')
            print(f"=================================== Training Linear Classifier with All "
                  f"features===================================")
            evaluate_linear_classifier(ten_fold=False, test_images=test_images, test_classes=test_classes,
                                       training_images=images, training_classes=classes, tolerance=tolerance,
                                       c_value=c_value, seed_value=seed_value, num_of_attrs=9690)

            # Import TOP SELECTED FEATURES from neural_attr_selection
            for i in TOP_SELECTED_FEATURES:
                images = pd.read_csv(f"{DATA_PATH}/neural/csv/x/x_train_gr_smpl_top_{i}.csv")
                classes = pd.read_csv(f"{DATA_PATH}/neural/csv/y/y_train_smpl_top_{i}.csv")
                test_images = pd.read_csv(f'{DATA_PATH}/neural/csv/x/x_test_gr_smpl_top_{i}.csv')
                test_classes = pd.read_csv(f'{DATA_PATH}/neural/csv/y/y_test_smpl_top_{i}.csv')
                print(f"=================================== Training Linear Classifier with {i} features"
                      f"===================================")
                evaluate_linear_classifier(ten_fold=False, test_images=test_images, test_classes=test_classes,
                                           training_images=images, training_classes=classes, tolerance=tolerance,
                                           c_value=c_value, seed_value=seed_value, num_of_attrs=i)


def run_MLP_task1(seed_value: int):
    for tolerance in TOLERANCE_VALUES:
        print(f"=================================== Iterating with tolerance of {tolerance} set"
              f" ===================================")
        for max_iter_value in MAX_ITERATION_VALUES:
            print(f"=================================== Iterating with max_iter_value of {max_iter_value} "
                  "set ===================================")
            for layer_size_value in LAYER_SIZES_VALUES:
                print(f"=================================== Iterating with layer_size_value of {layer_size_value} "
                      "set=================================")
                # Fit and test using training set, without then with ten fold:
                for tenfold in [False, True]:
                    images = pd.read_csv(f"{DATA_PATH}/x_train_gr_smpl.csv")
                    classes = pd.read_csv(f"{DATA_PATH}/y_train_smpl.csv")
                    test_images = images
                    test_classes = classes
                    print(f"=================================== Training MLP Classifier with All features,"
                          f" TenFold={tenfold} ===================================")
                    evaluate_MLP_classifier(ten_fold=tenfold, test_images=test_images, test_classes=test_classes,
                                            training_images=images, training_classes=classes, tolerance=tolerance,
                                            max_iterations=max_iter_value, layer_sizes=layer_size_value,
                                            seed_value=seed_value, num_of_attrs=9690)

                    # Import TOP SELECTED FEATURES from neural_attr_selection
                    for i in TOP_SELECTED_FEATURES:
                        images = pd.read_csv(f"{DATA_PATH}/neural/csv/x/x_train_gr_smpl_top_{i}.csv")
                        classes = pd.read_csv(f"{DATA_PATH}/neural/csv/y/y_train_smpl_top_{i}.csv")
                        test_images = images
                        test_classes = classes
                        print(
                            f"=================================== Training MLP Classifier with {i} features,"
                            f" TenFold={tenfold} ===================================")
                        evaluate_MLP_classifier(ten_fold=tenfold, test_images=test_images, test_classes=test_classes,
                                                training_images=images, training_classes=classes, tolerance=tolerance,
                                                max_iterations=max_iter_value, layer_sizes=layer_size_value,
                                                seed_value=seed_value, num_of_attrs=i)


def run_MLP_task3(seed_value: int):
    for tolerance in TOLERANCE_VALUES:
        print(f"=================================== Iterating with tolerance of {tolerance} "
              "set=================================")
        for max_iter_value in MAX_ITERATION_VALUES:
            print(f"================Iterating with max_iter_value of {max_iter_value} "
                  "set=================================")
            for layer_size_value in LAYER_SIZES_VALUES:
                print(f"================Iterating with layer_size_value of {layer_size_value}"
                      f" set=================================")
                images = pd.read_csv(f"{DATA_PATH}/x_train_gr_smpl.csv")
                classes = pd.read_csv(f"{DATA_PATH}/y_train_smpl.csv")
                test_images = pd.read_csv(f'{DATA_PATH}/x_test_gr_smpl.csv')
                test_classes = pd.read_csv(f'{DATA_PATH}/y_test_smpl.csv')
                print(f"=================================== Training MLP Classifier with All features"
                      f" ===================================")
                evaluate_MLP_classifier(ten_fold=False, test_images=test_images, test_classes=test_classes,
                                        training_images=images, training_classes=classes, tolerance=tolerance,
                                        max_iterations=max_iter_value, layer_sizes=layer_size_value,
                                        seed_value=seed_value, num_of_attrs=9690)

                # Import TOP SELECTED FEATURES from neural_attr_selection
                for i in TOP_SELECTED_FEATURES:
                    images = pd.read_csv(f"{DATA_PATH}/neural/csv/x/x_train_gr_smpl_top_{i}.csv")
                    classes = pd.read_csv(f"{DATA_PATH}/neural/csv/y/y_train_smpl_top_{i}.csv")
                    test_images = pd.read_csv(f'{DATA_PATH}/neural/csv/x/x_test_gr_smpl_top_{i}.csv')
                    test_classes = pd.read_csv(f'{DATA_PATH}/neural/csv/y/y_test_smpl_top_{i}.csv')
                    print(f"=================================== Training MLP Classifier with {i} features "
                          f"===================================")
                    evaluate_MLP_classifier(ten_fold=False, test_images=test_images, test_classes=test_classes,
                                            training_images=images, training_classes=classes, tolerance=tolerance,
                                            max_iterations=max_iter_value, layer_sizes=layer_size_value,
                                            seed_value=seed_value, num_of_attrs=i)


# Run all linear classifiers for a given seed value
def run_all_linear_classifier(seed_value: int):
    # Task 1 = "Using the provided training data sets, and the 10-fold cross validation"
    print("=====================TASK 1 - Training on training set, testing on training set=====================")
    run_linear_task1(seed_value)

    # Task 3 = "Repeat steps 1 and 2, this time using training and testing data sets instead of the cross
    # validation.That is, build the classifier using the training data set, and test the classifier using the
    # providedtest data set. Note the accuracy"
    print("=====================TASK 3 - Training on training set, testing on testing set=====================")
    run_linear_task3(seed_value)


# Run all linear classifiers for a given seed value
def run_all_MLP(seed_value: int):
    # Task 1 = "Using the provided training data sets, and the 10-fold cross validation"
    print("=====================TASK 1 - Training on training set, testing on training set=====================")
    run_MLP_task1(seed_value)

    # Task 3 = "Repeat steps 1 and 2, this time using training and testing data sets instead of the cross
    # validation.That is, build the classifier using the training data set, and test the classifier using the
    # providedtest data set. Note the accuracy"
    print("=====================TASK 3 - Training on training set, testing on testing set=====================")
    run_MLP_task3(seed_value)


def main():
    # Perform attribute selection on the data
    # neural_attr_selection.main()

    # Create output CSV file
    linear_df = DataFrame([['Type', 'Seed', 'Parameters (Number of features, Tolerance, c value, Ten Fold)',
                            'Accuracy', 'Precision', 'F_Score', 'Recall', "ROC_Area", "TP Rate", "FP Rate",
                            "Confusion_Matrix", "Classification_Report"]])
    linear_df.to_csv(f"{EVIDENCE_PATH}/linear.csv", header=True)
    mlp_df = DataFrame([['Type', 'Seed', 'Parameters (Number of features, Tolerance, Iterations, Ten Fold)',
                         'Accuracy', 'Precision', 'F_Score', 'Recall', "ROC_Area", "TP Rate", "FP Rate",
                         "Confusion_Matrix", "Classification_Report"]])
    mlp_df.to_csv(f"{EVIDENCE_PATH}/mlp.csv", header=True)

    # Loop through defined seed values
    for i in SEED_VALUES:
        print(f"=========================== Iterating with Seed value of {i} ===========================")
        print(f"=========================== Executing linear classifiers ===========================")
        # run_all_linear_classifier(i)
        print(f"=========================== Executing MLP ===========================")
        run_all_MLP(i)


if __name__ == "__main__":
    main()
