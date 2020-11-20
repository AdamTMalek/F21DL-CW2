from typing import Dict

import pandas as pd
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, accuracy_score, \
    average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from data import get_data_dir_path

from neural_attr_selection import TOP_SELECTED_FEATURES
DATA_PATH = get_data_dir_path()
SEED = 10

def get_scores(classifier, ten_fold: bool, images: DataFrame, classes: DataFrame) -> Dict:
    result = {}
    if ten_fold:
        prediction = cross_val_predict(classifier, images, classes, cv=10, n_jobs=-1)
        probability = cross_val_predict(classifier, images, classes, cv=10, n_jobs=-1, method='predict_proba')
        # score = cross_val_score(classifier, images, classes, cv=10, n_jobs=-1)
    else:
        prediction = classifier.predict(images)
        probability = classifier.predict_proba(images)
        # score = classifier.score(images, classes)
    print(prediction)
    result["accuracy"] = accuracy_score(classes, prediction)
    result["precision"] = precision_score(classes, prediction, average='weighted')
    # result['average_precision'] = average_precision_score(classes, prediction, average='weighted', pos_label=0)
    result["f_score"] = f1_score(classes, prediction, average='weighted')
    result["recall"] = recall_score(classes, prediction, average='weighted')
    # result['fpr'], result['tpr'], _ = roc_curve(y_true=classes, y_score=score)
    result["roc_area"] = roc_auc_score(y_true=classes, y_score=probability, average='weighted', multi_class='ovo')
    result["confusion_matrix"] = confusion_matrix(y_true=classes, y_pred=prediction,
                                                  labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    result["classification_report"] = classification_report(y_true=classes, y_pred=prediction)
    return result


def print_scores(scores: Dict) -> None:
    print(f'Accuracy: {scores["accuracy"]}')
    print(f'Precision: {scores["precision"]}')
    # print(f'Average precision: {scores["average_precision"]}')        #Used for One v Rest
    print(f'F score: {scores["f_score"]}')
    print(f'Recall: {scores["recall"]}')
    # print(f'True Positive: {scores["tpr"]}')                          #Use for One v Rest
    # print(f'False Positive: {scores["fpr"]}')                         #Use for One v Rest
    print(f'ROC area: {scores["roc_area"]}')
    print(scores["confusion_matrix"])
    print(scores["classification_report"])


def evaluate_linear_classifier(ten_fold: bool, test_images: DataFrame, test_classes: DataFrame, training_images: DataFrame, training_classes: DataFrame):
    # Normalise training data set
    training_images = training_images / 255
    training_classes = training_classes.values.ravel()

    # Linear
    linear = LogisticRegression(max_iter=15000,
                                tol=0.5,
                                C=1.0,
                                random_state=SEED).fit(training_images, training_classes)

    # Noramlise test data set
    test_images = test_images / 255
    test_classes = test_classes.values.ravel()

    # Get scores and results
    linear_result = get_scores(linear, ten_fold=ten_fold, images=test_images, classes=test_classes)

    print_scores(linear_result)

def evaluate_MLP_classifier(ten_fold: bool, test_images: DataFrame, test_classes: DataFrame, training_images: DataFrame, training_classes: DataFrame):
    # Normalise training data set
    training_images = training_images / 255
    training_classes = training_classes.values.ravel()

    # # MLP (THIS IS VEEEERY SLOW)
    mlp_classifier = MLPClassifier(solver='lbfgs',  #DO NOT REMOVE THIS LINE UNLESS YOU KNOW WHAT YOU ARE DOING
                                   activation='logistic',
                                   learning_rate='adaptive',
                                   max_iter=15000,
                                   random_state=SEED).fit(test_images, test_classes.values.ravel())

    # Noramlise test data set
    test_images = test_images / 255
    test_classes = test_classes.values.ravel()

    # Get scores and results
    mlp_result = get_scores(mlp_classifier, ten_fold, images=training_images, classes=training_classes)

    print_scores(mlp_result)

#Mess at the minute, but does the trick
def run_all_linear_classifier():
    #Task 1 = "Using the provided training data sets, and the 10-fold cross validation"
    print("============================TASK 1 - Training on training set, testing on training set=================================\n\n\n\n")
    #Fit and test using training set, without then with ten fold:
    for tenfold in [False,True]:
        images = pd.read_csv(f"{DATA_PATH}/x_train_gr_smpl.csv")
        classes = pd.read_csv(f"{DATA_PATH}/y_train_smpl.csv")
        test_images = images #pd.read_csv(f'{DATA_PATH}/x_test_gr_smpl.csv')
        test_classes = classes #pd.read_csv(f'{DATA_PATH}/y_test_smpl.csv')
        print(f"=================================== Training Linear Classifier with All features, TenFold={tenfold}===================================")
        evaluate_linear_classifier(ten_fold=tenfold, test_images=test_images, test_classes=test_classes, training_images=images, training_classes=classes)

        #Import TOP SELECTED FEATURES from neural_attr_selection
        for i in TOP_SELECTED_FEATURES:
            images = pd.read_csv(f"{DATA_PATH}/neural/csv/x/x_train_gr_smpl_top_{i}.csv")
            classes = pd.read_csv(f"{DATA_PATH}/neural/csv/y/y_train_smpl_top_{i}.csv")
            test_images = images #pd.read_csv(f'{DATA_PATH}/neural/csv/x/x_test_gr_smpl_top_{i}.csv')
            test_classes = classes #pd.read_csv(f'{DATA_PATH}/neural/csv/y/y_test_smpl_top_{i}.csv')
            print(f"=================================== Training Linear Classifier with {i} features, TenFold={tenfold}===================================")
            evaluate_linear_classifier(ten_fold=tenfold, test_images=test_images, test_classes=test_classes, training_images=images, training_classes=classes)




    #Task 3 = "Repeat steps 1 and 2, this time using training and testing data sets instead of the cross validation.That is, build the classifier using the training data set, and test the classifier using the providedtest data set. Note the accuracy"
    print("============================TASK 3 - Training on training set, testing on testing set=================================\n\n\n\n")
    #Fit and test using training set, without then with ten fold:
    images = pd.read_csv(f"{DATA_PATH}/x_train_gr_smpl.csv")
    classes = pd.read_csv(f"{DATA_PATH}/y_train_smpl.csv")
    test_images = pd.read_csv(f'{DATA_PATH}/x_test_gr_smpl.csv')
    test_classes = pd.read_csv(f'{DATA_PATH}/y_test_smpl.csv')
    print(f"=================================== Training Linear Classifier with All features===================================")
    evaluate_linear_classifier(ten_fold=False, test_images=test_images, test_classes=test_classes, training_images=images, training_classes=classes)

    #Import TOP SELECTED FEATURES from neural_attr_selection
    for i in TOP_SELECTED_FEATURES:
        images = pd.read_csv(f"{DATA_PATH}/neural/csv/x/x_train_gr_smpl_top_{i}.csv")
        classes = pd.read_csv(f"{DATA_PATH}/neural/csv/y/y_train_smpl_top_{i}.csv")
        test_images = pd.read_csv(f'{DATA_PATH}/neural/csv/x/x_test_gr_smpl_top_{i}.csv')
        test_classes = pd.read_csv(f'{DATA_PATH}/neural/csv/y/y_test_smpl_top_{i}.csv')
        print(f"=================================== Training Linear Classifier with {i} features===================================")
        evaluate_linear_classifier(ten_fold=False, test_images=test_images, test_classes=test_classes, training_images=images, training_classes=classes)


def main():
    run_all_linear_classifier()

if __name__ == "__main__":
    main()








# def ten_fold_classifier(images: DataFrame, classes: DataFrame):
#     # Data
#     images = images / 255
#     classes = classes.values.ravel()

#     # Linear
#     linear_classifier = LogisticRegression(max_iter=15000,
#                                            tol=0.0001,
#                                            C=10.0,
#                                            random_state=SEED)
#     linear_result = get_scores(linear_classifier, True, images, classes)

#     # MLP
#     mlp_classifier = MLPClassifier(hidden_layer_sizes=(150, 100, 50, 25),
#                                    max_iter=15000,
#                                    random_state=SEED)
#     mlp_result = get_scores(mlp_classifier, True, images, classes)
    
#     # Print
#     print("=================================== Ten fold Linear Classifier ===================================")
#     print_scores(linear_result)
#     print("==================================================================================================")
#     print("=================================== Ten fold MLP Classifier ======================================")
#     print_scores(mlp_result)
#     print("==================================================================================================")



