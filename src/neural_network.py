import pandas as pd
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, accuracy_score, \
    average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from data import get_data_dir_path

DATA_PATH = get_data_dir_path()
SEED = 9


def get_scores(classifier, ten_fold: bool, images, classes):
    result = {}
    if ten_fold:
        prediction = cross_val_predict(classifier, images, classes, cv=10, n_jobs=-1)
        score = cross_val_score(classifier, images, classes, cv=10, n_jobs=-1)
    else:
        prediction = classifier.predict(classifier, images, classes, cv=10, n_jobs=-1)
        score = classifier.score(classifier, images, classes, cv=10, n_jobs=-1)
    result['accuracy'] = accuracy_score(classes, prediction, normalize=True)
    result['precision'] = precision_score(classes, prediction)
    result['average_precision'] = average_precision_score(classes, prediction)
    result['f_score'] = f1_score(classes, prediction)
    result['recall'] = recall_score(classes, prediction)
    result['fpr'], result['tpr'], result['thresholds'] = roc_curve(y_true=classes, y_score=score)
    result['roc_area'] = roc_auc_score(y_true=classes, y_score=score)
    result['confusion_matrix'] = confusion_matrix(y_true=classes, y_pred=prediction)
    result['classification_report'] = classification_report(y_true=classes, y_pred=prediction)
    return result


def print_scores(scores):
    print(f'Accuracy: {scores["accuracy"]}')
    print(f'Precision: {scores["precision"]}')
    print(f'Average precision: {scores["average_precision"]}')
    print(f'F score: {scores["f_score"]}')
    print(f'Recall: {scores["recall"]}')
    print(f'True Positive: {scores["tpr"]}')
    print(f'False Positive: {scores["fpr"]}')
    print(f'ROC area: {scores["roc_area"]}')
    print(scores["confusion_matrix"])
    print(scores["classification_report"])


def training_set_classifier(test_images: DataFrame, test_classes: DataFrame, images: DataFrame, classes: DataFrame):
    # Data
    images = images / 255
    classes = classes.values.ravel()

    # Linear
    linear = LogisticRegression(max_iter=15000, tol=0.0001, C=10.0, random_state=SEED).fit(test_images,
                                                                                        test_classes.values.ravel())
    linear_result = get_scores(linear, False, images, classes)

    # MLP
    mlp_classifier = MLPClassifier(activation='relu', learning_rate_init=1, max_iter=15000,
                                   random_state=SEED).fit(test_images, test_classes.values.ravel())
    mlp_result = get_scores(mlp_classifier, False, images, classes)

    # Print
    # Print
    print("=================================== Training Linear Classifier ===================================")
    print_scores(linear_result)
    print("==================================================================================================")
    print("=================================== Training MLP Classifier ======================================")
    print_scores(mlp_result)
    print("==================================================================================================")


def ten_fold_classifier(images: DataFrame, classes: DataFrame):
    # Data
    images = images / 255
    classes = classes.values.ravel()

    # Linear
    linear_classifier = LogisticRegression(max_iter=15000, tol=0.0001, C=10.0, random_state=SEED)
    linear_result = get_scores(linear_classifier, True, images, classes)

    # MLP
    mlp_classifier = MLPClassifier(max_iter=15000, random_state=SEED)
    mlp_result = get_scores(mlp_classifier, True, images, classes)

    # Print
    print("=================================== Ten fold Linear Classifier ===================================")
    print_scores(linear_result)
    print("==================================================================================================")
    print("=================================== Ten fold MLP Classifier ======================================")
    print_scores(mlp_result)
    print("==================================================================================================")


def print_scores(accuracy, average_precision, tp, fp, precision, recall, f_score, roc):
    print(f'Accuracy score: {accuracy}')
    print(f'Precision score: {precision}')
    print(f'Average precision score: {average_precision}')
    print(f'F-Score: {f_score}')
    print(f'Recall: {recall}')
    print(f'ROC area: {roc}')
    print(f'True Positive: {tp}')
    print(f'False Positive: {fp}')


def main():
    test_images = pd.read_csv(f'{DATA_PATH}/x_test_gr_smpl.csv')
    test_classes = pd.read_csv(f'{DATA_PATH}/y_test_smpl.csv')
    images = pd.read_csv(f"{DATA_PATH}/x_train_gr_smpl.csv")
    classes = pd.read_csv(f"{DATA_PATH}/y_train_smpl.csv")
    ten_fold_classifier(images=images, classes=classes)
    training_set_classifier(test_images=test_images, test_classes=test_classes, images=images, classes=classes)


if __name__ == "__main__":
    main()
