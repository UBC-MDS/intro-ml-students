from hashlib import sha1
import pandas as pd
import pytest
import altair
import sys
import numpy as np


def test_1_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(answer.lower().encode('utf8')).hexdigest() == "25aeefb67cd1b85d531b5a8c34c98f807c93bf6f", "Your answer is incorrect. Please try again."
    return("Success")

def test_1_2_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,3)).encode('utf8')).hexdigest() == "76108e5874d65b0aec9d1e103541ed6c8d2df69f", "Your answer is incorrect. Please try again."
    return("Success")

def test_1_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(answer.lower().encode('utf8')).hexdigest() == "4d8235d83eeac8c909d966beac6de30bdaab6012", "Your answer is incorrect. Are you examining the models?"
    return("Success")

def test_1_4_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,3)).encode('utf8')).hexdigest() == "e8dc057d3346e56aed7cf252185dbe1fa6454411", "Your answer is incorrect. Please try again."
    return("Success")

def test_1_4_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,3)).encode('utf8')).hexdigest() == "9588c3fcb43fc86f5ac79164cedf59e8e7b9e7ec", "Your answer is incorrect. Please try again."
    return("Success")

def test_1_4_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,3)).encode('utf8')).hexdigest() == "959041c33324191578173f21a203e93aa2c2b431", "Your answer is incorrect. Please try again."
    return("Success")

def test_1_5_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,3)).encode('utf8')).hexdigest() == "856b62aa687c0aa2b0deb2980b3dd887b3c93ff8", "Your answer is incorrect. Please try again."
    return("Success")

def test_1_5_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,3)).encode('utf8')).hexdigest() == "fbccebab4a05483ba1b7ce5c12688ac8f69ec024", "Your answer is incorrect. Please try again."
    return("Success")

def test_1_5_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,3)).encode('utf8')).hexdigest() == "18739de0047c291fd062b8733600651cf952d304", "Your answer is incorrect. Please try again."
    return("Success")

def test_1_6(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert len(answer) == 2, "The number of correct answers in incorrect. Please select all the correct answers."
    assert sha1(str(sorted(answer)).encode('utf8')).hexdigest() == "98d38510ecf7f95b1cee57c3c03cb16044223b36", "Your answers are incorrect. Think about what we would care about in this case."
    return("Success")

def test_1_7(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(answer.lower().encode('utf8')).hexdigest() == "a344ee4892b7cfc3923da199939806e49038026c", "Your answer is incorrect. Are you examining the models correctly?."
    return("Success")

def test_2_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    answer = list(answer)
    assert sha1(str(answer[0]).encode('utf8')).hexdigest() == "1c8e6d9301b25a912093090c7561436e71d7a544", "The value for not excited is incorrect. Are you using the count function?"
    assert sha1(str(answer[1]).encode('utf8')).hexdigest() == "d69729e9779952a73e8ef1e71858b08a92a0fa8b", "The value for excited is incorrect. Are you using the count function?"
    return("Success")

def test_2_2(answer1,answer2,answer3,answer4):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer3 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer3 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer1.shape == (7000, 11), "The dimensions of the training set is incorrect. Are you splitting correctly?"
    assert answer2.shape == (3000, 11), "The dimensions of the test set is incorrect. Are you splitting correctly"
    assert answer3.shape == (7000,), "The dimensions of the training set is incorrect. Are you splitting correctly? Are you using single brackets?"
    assert answer4.shape == (3000,), "The dimensions of the test set is incorrect. Are you splitting correctly? Are you using single brackets?"
    assert 'Exited' not in answer1.columns, "Make sure you are dropping the target column from the training X dataset."
    assert 'Exited' not in answer2.columns, "Make sure you are dropping the target column from the testing X dataset."
    return("Success")

def test_2_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    answer = round(answer.sum(), 1)
    assert answer['test_accuracy'] >=  3.3, "The range of your test accuracy is incorrect. Are you fitting the model properly?"
    assert answer['train_accuracy'] >=  3.3, "The range of your training accuracy is incorrect. Are you fitting the model properly?"
    assert answer['test_f1'] >= 1, "The range of your test f1 scores is incorrect. Are you fitting the model properly?"
    assert answer['train_f1'] >= 1, "The range of your training f1 scores is incorrect. Are you fitting the model properly?"
    assert answer['test_recall'] >= 1, "The range of your test recall scores is incorrect. Are you fitting the model properly?"
    assert answer['train_recall'] >= 1, "The range of your training recall scores is incorrect. Are you fitting the model properly?"
    assert answer['test_precision'] >= 1, "The range of your test precision scores is incorrect. Are you fitting the model properly?"
    assert answer['train_precision'] >= 1, "The range of your training precision scores is incorrect. Are you fitting the model properly?"
    return("Success")

def test_2_4(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    answer = [round(x,1) for x in list(answer)][2:10]
    assert min(answer) >= 0.2 and max(answer) <= 0.70, "Your values are incorrect. Are you taking the mean of each column?"
    return("Success")

def test_2_5_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert len(answer) == 8, "The number of numeric features is incorrect. Are you analysing the data correctly?"
    answer = [x.lower() for x in sorted(list(answer))]
    assert sha1(str(answer).encode('utf8')).hexdigest() == "e2ac9f66b793912611aa37b7a9d4e5f35b9b10f8", "The numerical features are incorrect. Are you analyzing the data correctly?"
    return("Success")

def test_2_5_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    answer = [x.lower() for x in sorted(list(answer))]
    assert len(answer) == 1, "The number of features to drop is incorrect. Are you analysing the data correctly?"
    assert sha1(str(answer).encode('utf8')).hexdigest() == "4e16a2da8bcfed9b2787a9946e5a311accabcc23", "The feature to drop is incorrect. Are you analyzing the data correctly?"
    return("Success")

def test_2_5_4(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert len(answer) == 1, "The number of binary features is incorrect. Are you analysing the data correctly?"
    answer = [x.lower() for x in list(answer)]
    assert sha1(str(answer).encode('utf8')).hexdigest() == "82cde58cbe55c13ed57d26c474d181e62ca88900", "The binary features are incorrect. Are you analyzing the data correctly?"
    return("Success")

def test_2_6(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sorted(list(answer.columns)) == ['fit_time','score_time','test_accuracy','test_f1','test_precision','test_recall','train_accuracy','train_f1','train_precision','train_recall'], "\
    Your dataframe is missing a some columns. Are you returning all the metrics?"
    answer = round(answer.sum(), 1)
    assert answer['test_accuracy'] >=  4.3, "The range of your test accuracy is incorrect. Are you fitting the model properly?"
    assert answer['train_accuracy'] >=  5.0, "The range of your training accuracy is incorrect. Are you fitting the model properly?"
    assert answer['test_precision'] >= 3.8, "The range of your test precision scores is incorrect. Are you fitting the model properly?"
    assert answer['train_precision'] >= 5.0, "The range of your training precision scores is incorrect. Are you fitting the model properly?"
    assert answer['test_recall'] >= 2.2, "The range of your test recall scores is incorrect. Are you fitting the model properly?"
    assert answer['train_recall'] >= 5.0, "The range of your training recall scores is incorrect. Are you fitting the model properly?"
    assert answer['test_f1'] >= 2.8, "The range of your test f1 scores is incorrect. Are you fitting the model properly?"
    assert answer['train_f1'] >= 5.0, "The range of your training f1 scores is incorrect. Are you fitting the model properly?"
    return("Success")

def test_2_7(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    answer = [round(x,1) for x in list(answer)][2:10]
    assert min(answer) >= 0.4 and max(answer) <= 1, "Your values are incorrect. Are you taking the mean of each column?"
    return("Success")

def test_2_8(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    answer = round(answer.sum(), 1)
    assert answer['test_accuracy'] >=  4.3, "The range of your test accuracy is incorrect. Are you fitting the model properly?"
    assert answer['train_accuracy'] >=  5.0, "The range of your training accuracy is incorrect. Are you fitting the model properly?"
    assert answer['test_precision'] >= 3.8, "The range of your test precision scores is incorrect. Are you fitting the model properly?"
    assert answer['train_precision'] >= 5.0, "The range of your training precision scores is incorrect. Are you fitting the model properly?"
    assert answer['test_recall'] >= 2.1, "The range of your test recall scores is incorrect. Are you fitting the model properly?"
    assert answer['train_recall'] >= 5.0, "The range of your training recall scores is incorrect. Are you fitting the model properly?"
    assert answer['test_f1'] >= 2.0, "The range of your test f1 scores is incorrect. Are you fitting the model properly?"
    assert answer['train_f1'] >= 4.0, "The range of your training f1 scores is incorrect. Are you fitting the model properly?"
    return("Success")

def test_2_9(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    answer = [round(x,1) for x in list(answer)][2:10]
    assert min(answer) >= 0.4 and max(answer) <= 1, "Your values are incorrect. Are you taking the mean of each column?"
    return("Success")

def test_2_10(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert len(answer) == 3, "The number of correct answers is incorrect. Are you analyzing the results carefully?"
    answer = [x.lower() for x in answer]
    assert sha1(str(sorted(answer)).encode('utf8')).hexdigest() == "a365da2abbe31b0e8a8400ad876f0b619dd65ab9", "Your answer is incorrect. Are you analyzing the results carefully?"
    return("Success")

def test_2_11(answer1,answer2):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert 'RandomForestClassifier' in str(answer1.get_params()['steps']), "Your rf_pipeline is incomplete. Please specify a classifier."
    assert 'balanced' in str(answer1.get_params()['steps']), "Your rf_pipeline is incomplete. Please specify the class weights."
    assert  list(answer2.param_distributions.keys()) == ['randomforestclassifier__n_estimators', 'randomforestclassifier__max_depth'], "Make sure you are specifying the distribution of the parameters."
    return("Success")

def test_2_12_1_old(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer['randomforestclassifier__max_depth'] in range(2, 20), "Your value for max depth is incorrect. Are you using the .best_params_ function?"
    assert answer['randomforestclassifier__n_estimators'] in range(10, 300), "Your value for number of estimators is incorrect. Are you using the .best_params_ function?"
    return("Success")


def test_2_12_1(answer, answer1):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.best_params_['randomforestclassifier__max_depth'] == answer1['randomforestclassifier__max_depth'], "Your value for max depth is incorrect. Are you using the .best_params_ function?"
    assert answer.best_params_['randomforestclassifier__n_estimators'] == answer1['randomforestclassifier__n_estimators'], "Your value for number of estimators is incorrect. Are you using the .best_params_ function?"
    return("Success")


def test_2_13_1(answer, answer1, X_train, y_train):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert round(answer.score(X_train, y_train), 2) == round(answer1, 2), "Your training score is incorrect. Are you fitting and scoring the model properly?"
    return("Success")


def test_2_13_2(answer, answer1, X_test, y_test):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert round(answer.score(X_test, y_test), 2) == round(answer1, 2), "Your testing score is incorrect. Are you fitting and scoring the model properly?"
    return("Success")


def test_2_14():
    assert 'sklearn' in sys.modules, "Make sure you are importing 'plot_confusion_matrix' and 'classification_report' from the sklearn module."
    return("Success")

def test_2_15(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert str(type(answer)) == "<class 'sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay'>", "Make sure you are generating a confusion matrix plot"
    return("Success")

def check(value):
    if 0.60 <= value <= 0.80 and round(value,2)==value:
        return True
    return False

def check2(value):
    if 0.60 <= value <= 0.90 and round(value,2)==value:
        return True
    return False

def test_2_16_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert check(answer), "The recall value for exicted is incorrect. Are you analyzing the results correctly?"
    return("Success")

def test_2_16_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert check2(answer), "The value for the weighted average is incorrect. Are you analyzing the results correctly?"
    return("Success")

def test_2_16_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer).encode('utf8')).hexdigest() == "8290abc6c261e044710e7d616082ab51cb377262", "The number of customers is incorrect. Are you analyzing the results correctly?"
    return("Success")

def test_2_17(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1((answer.lower() + 'k').encode('utf8')).hexdigest() == "d5658db4705ff5c6ad3f152c4393f61ac86bd27f", "Your answer is incorrect. How many of these are correct?"
    return("Success")

def test_3_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert "('standardscaler', StandardScaler())" in str(answer.get_params()['steps']), "Make sure you are using the standard scaler in your model"
    assert "('svr', SVR())" in str(answer.get_params()['steps']), "Make sure you are using the support vector regressor SVR()"
    return("Success")

def test_3_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert 'mape' in str(answer), "Make sure your are passing the mape function to the make_scorer function."
    assert 'greater_is_better=False' in str(answer), "Don't for get to specify the greater_is_better argument."
    return("Success")

def test_3_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert str(type(answer)) == "<class 'dict'>", "Make sure your answer is a dictionary."
    assert sorted(list(answer.keys())) == ['mape_scorer','neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_square','r2'] or sorted(list(answer.keys())) == ['mape_scorer','neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_square_error','r2'], "Your dictionary is missing some keys. Make sure you are adding all the metrics."
    assert 'mape' in str(answer['mape_scorer']), "Make sure your are passing the make_scorer function to the dictionary."
    assert 'greater_is_better=False' in str(answer['mape_scorer']), "Make sure your are passing the make_scorer function to the dictionary."
    return("Success")

def test_3_4(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    answer = round(answer.sum(), 1)
    assert answer['test_neg_mean_squared_error'].sum().round(0) == -419.0, "The range of your test negative mean squared error is incorrect. Are you fitting the model properly?"
    assert answer['train_neg_mean_squared_error'].sum().round(0) == -399.0, "The range of your training negative mean squared error is incorrect. Are you fitting the model properly?"
    assert answer['test_neg_root_mean_square_error'].sum().round(0) == -44.0, "The range of your test negative root mean squared error is incorrect. Are you fitting the model properly?"
    assert answer['train_neg_root_mean_square_error'].sum().round(0) == -44.0, "The range of your training negative root mean squared error is incorrect. Are you fitting the model properly?"
    assert answer['test_r2'].sum().round(0) == 3.0, "The range of your test r2 is incorrect. Are you fitting the model properly?"
    assert answer['train_r2'].sum().round(0) == 3.0, "The range of your training r2 is incorrect. Are you fitting the model properly?"
    assert answer['test_mape_scorer'].sum().round(0) == -89.0, "The range of your test mape scorer is incorrect. Are you fitting the model properly?"
    assert answer['train_mape_scorer'].sum().round(0) == -84.0, "The range of your training mape scorer is incorrect. Are you fitting the model properly?"
    return("Success")

def test_3_5(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    answer = [round(x,1) for x in list(answer)][2:12]
    assert min(answer) == -83.7 and max(answer) == 0.6, "Your values are incorrect. Are you taking the mean of each column?"
    return("Success")

