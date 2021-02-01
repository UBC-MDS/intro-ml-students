from hashlib import sha1
import pandas as pd
import pytest
import altair
import sys
import numpy as np


def test_1_1(answer1,answer2):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer1.shape == (12545, 10), "The dimensions of training set is incorrect. Are you splitting correctly?"
    assert answer2.shape == (3137, 10), "The dimensions of the test set is incorrect. Are you splitting correctly?"
    assert list(answer1.loc[1285]) == [46, '5', 121124, 13, 1, 1, 15024, 0, 40, '>50K'], "Make sure you are setting your random state to 123."
    assert list(answer2.loc[8024]) == [48, '3', 213140, 9, 1, 1, 0, 0, 80, '<=50K'], "Make sure you are setting your random state to 123"
    return("Success")

def test_1_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(answer.encode('utf8')).hexdigest() == "5c6bfa705da99e52ea42d95cd6c5fc8addb8a531", "Your answer is incorrect. Are you examining the data types correctly?"
    return("Success")

def test_1_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(answer.encode('utf8')).hexdigest() == "5bab61eb53176449e25c2c82f172b82cb13ffb9d", "Your answer is incorrect. Are you using the .unique function?"
    return("Success")

def test_1_4(answer1,answer2):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert "?" not in answer1.values, "Make sure you are replacing all the required value with NAN"
    assert "?" not in answer2.values, "Make sure you are replacing all the required value with NAN"
    return("Success")

def test_1_5(answer1,answer2):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert str(answer1.dtypes['workclass']) == 'float64', "Make sure you are changing the data type of the 'workclass' column."
    assert str(answer2.dtypes['workclass']) == 'float64', "Make sure you are changing the data type of the 'workclass' column."
    return("Success")

def test_1_6(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.shape == (8, 9), "The dimensions of your solution is incorrect. Are you using the describe function?"
    assert 'std' in list(answer.index), "Your solution is missing some values. Are you using the describe function?"
    assert 'education_num' in list(answer.columns), "Your solution is missing some columns. Are you using the correct dataframe?"
    assert list(answer.iloc[7]) == [90.0, 6.0, 1484705.0, 16.0, 6.0, 1.0, 99999.0, 4356.0, 99.0], "Your solution is missing some values. Are you using the describe function?"
    return("Success")

def test_1_7_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,2)).encode('utf8')).hexdigest() == "18cb317f5646550c18aedf3bfef693d5e119ac01", "Your answer is incorrect. The mean function may be useful here or obtain the values from the train_stats."
    return("Success")

def test_1_8(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(answer.encode('utf8')).hexdigest() == "5dc56b9aab61867257a3c1bd7c786c9410d38cd2", "Your answer is incorrect. Are you examining the plots properly?"
    return("Success")

def test_1_9(answer1,answer2,answer3,answer4):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer3 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer3 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer1.shape == (12545, 9), "The dimensions of the training set is incorrect. Are you splitting correctly?"
    assert answer2.shape == (3137, 9), "The dimensions of the test set is incorrect. Are you splitting correctly"
    assert answer3.shape == (12545,), "The dimensions of the training set is incorrect. Are you splitting correctly? Are you using single brackets?"
    assert answer4.shape == (3137,), "The dimensions of the test set is incorrect. Are you splitting correctly? Are you using single brackets?"
    assert 'income' not in list(answer1.columns), "Make sure the target variable is not part of your X dataset."
    return("Success")

def test_1_10(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1((answer + 'i').encode('utf8')).hexdigest() == "fe564f789dc654a770ed03187c2200c93ee7be3d", "Your answer is incorrect. Can these models handle missing values?"
    return("Success")

def test_2_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.shape == (5, 4), "The dimensions of you solution is incorrect. Are you setting up the model correctly?"
    assert sorted(list(answer.columns)) == ['fit_time', 'score_time', 'test_score', 'train_score'], "Your dataframe contains the incorrect columns. Are you setting up the mocel correctly?"
    assert min(answer['test_score']) > 0 and max(answer['test_score']) < 1, "The range of your test scores is incorrect. Are you fitting the model properly?"
    assert min(answer['train_score']) > 0 and max(answer['train_score']) < 1, "The range of your training scores is incorrect. Are you fitting the model properly?"
    return("Success")

def test_2_2(answer1,answer2,answer3):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer3 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer1.strategy == 'median', "Make sure you are imputing using the median strategy"
    assert not np.isnan(answer2).any(), "Your dataframe contains missing values. Are you imputing properly?"
    assert not np.isnan(answer3).any(), "Your dataframe contains missing values. Are you imputing properly?"
    return("Success")

def test_2_4(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.shape == (5, 4), "The dimensions of you solution is incorrect. Are you setting up the model correctly?"
    assert sorted(list(answer.columns)) == ['fit_time', 'score_time', 'test_score', 'train_score'], "Your dataframe contains the incorrect columns. Are you setting up the mocel correctly?"
    assert min(answer['test_score']) > 0 and max(answer['test_score']) < 1, "The range of your test scores is incorrect. Are you fitting the model properly?"
    assert min(answer['train_score']) > 0 and max(answer['train_score']) < 1, "The range of your training scores is incorrect. Are you fitting the model properly?"
    return("Success")

def test_2_5(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1((answer + 'u').encode('utf8')).hexdigest() == "bbb8328c16f85703a882da420cc50f1df34c4868", "Your answer is incorrect. When is it okay to split the data?"
    return("Success")

def test_3_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.shape == (4, 4), "The dimensions of your solution is incorrect. Are you building the pipeline correctly?"
    assert sorted(list(answer.columns)) == ['mean_fit_time (s)','mean_score_time (s)','mean_train_accuracy','mean_validation_accuracy'], "Your solution contains incorrect columns. Are you specifying the pipeline correctly?"
    assert min(answer['mean_validation_accuracy']) > 0 and max(answer['mean_validation_accuracy']) < 1, "The range of your validaton scores is incorrect. Are you fitting the model properly?"
    assert min(answer['mean_train_accuracy']) > 0 and max(answer['mean_train_accuracy']) < 1, "The range of your training scores is incorrect. Are you fitting the model properly?"
    return("Success")

def test_3_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(answer.lower().encode('utf8')).hexdigest() == "04bb595b38a8f1345ce4d0d444b3c1d9c1b5d853", "Your solution is incorrect. Are you examining the results closely?"
    return("Success")

def test_3_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(answer.lower().encode('utf8')).hexdigest() == "19aaa8e3aa6d0eb8fff430cb6b72ba7a3fb83f67", "Your solution is incorrect. When does a model overfit?"
    return("Success")

def test_4_1():
    assert 'sklearn' in sys.modules, "Make sure you are importing 'GridSearchCV' and  'RandomizedSearchCV' from the sklearn module."
    return("Success")

def test_4_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.get_params()['cv'] == 5, "Make sure you are using 5-fold cross validation."
    assert 'knn' in str(answer.get_params()['estimator__steps']).split(',')[4], "Make sure you are using KNN in your pipeline steps."
    assert answer.get_params()['return_train_score'] == True, "Make sure you are returning the training score"
    assert str(list(answer.get_params()['estimator'])) == "[SimpleImputer(strategy='median'), StandardScaler(), KNeighborsClassifier()]", "Make sure you defining proper steps for the pipeline."
    return("Success")

def test_4_3_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer).encode('utf8')).hexdigest() == "472b07b9fcf2c2451e8781e944bf5f77cd8457c8", "Your solution is incorrect. Are you examining the results closely?"
    return("Success")

def test_4_3_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,2)).encode('utf8')).hexdigest() == "48026248575be074288e0b8334c8383a52f12906", "Your solution is incorrect. Are you examining the results closely?"
    return("Success")

def test_4_3_3(answer):
    assert sha1(str(answer).encode('utf8')).hexdigest() == "88b33e4e12f75ac8bf792aebde41f1a090f3a612", "Your solution is incorrect. Are you examining the results closely?"
    return("Success")

def test_4_4(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.get_params()['cv'] == 5, "Make sure you are using 5-fold cross validation."
    assert answer.get_params()['return_train_score'] == True, "Make sure you are returning the training score"
    assert str(list(answer.get_params()['estimator'])) == "[SimpleImputer(strategy='median'), StandardScaler(), SVC()]", "Make sure you defining proper steps for the pipeline."
    return("Success")

def test_4_5_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer['svc__gamma']).encode('utf8')).hexdigest() == "180505679cfe0cca79bae51fdda0296b7cd9c493", "Your solution is incorrect. Are you examining the results closely?"
    assert sha1(str(answer['svc__C']).encode('utf8')).hexdigest() == "b1d5781111d84f7b3fe45a0852e59758cd7a87e5", "Your solution is incorrect. Are you examining the results closely?"
    return("Success")


def test_4_5_2(answer1,answer2):
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer1.best_score_ == answer2, "Your solution is incorrect. Are you examining the results closely?"
    return("Success")

def test_4_6(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer).encode('utf8')).hexdigest() == "88b33e4e12f75ac8bf792aebde41f1a090f3a612", "Your answer is incorrect. Please try again."
    return("Success")

def test_5_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,2)).encode('utf8')).hexdigest() == "db96e1b97007bb15354e8b383cfb6dabe236c3fb", "Your solution is incorrect. Are you using the best model from 4 and scoring on X_train and y_train."
    return("Success")

def test_5_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,2)).encode('utf8')).hexdigest() == "6a54930730e9aa59489ecac1e78415ccd9053259", "Your solution is incorrect. Make sure you are scoring the best model from 4.4 on X_test and y_test."
    return("Success")

# +



