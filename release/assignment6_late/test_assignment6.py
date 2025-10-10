from hashlib import sha1
import pandas as pd
import pytest
import altair
import sys
import numpy as np


def test_1_1(answer1,answer2):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer1.shape == (31826, 15), "The dimensions of training set is incorrect. Are you splitting correctly?"
    assert answer2.shape == (7957, 15), "The dimensions of the test set is incorrect. Are you splitting correctly?"
    assert list(answer1.loc[14077])[0:4] == [7597, 'Bao Yingying', 'F', 24.0], "Make sure you are setting your random state to 123."
    assert list(answer2.loc[61002])[0:4] == [31253, 'Bohuslav Ebermann', 'M', 27.0], "Make sure you are setting your random state to 123"
    return("Success")

def test_1_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer).encode('utf8')).hexdigest() == "382012258ba94af8f99d0113ad9a907154558163", "Your answer is incorrect. Are you using the shape function?"
    return("Success")

def test_1_3_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(float(answer)).encode('utf8')).hexdigest() == "c23f7da7070511444ebc75875fd9d202b5dd13cf", "Your answer is incorrect. The describe function may be useful here."
    return("Success")

def test_1_4(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer).encode('utf8')).hexdigest() == "0ade7c2cf97f75d009975f4d720d1fa6c19f4897", "Your answer is incorrect. Are you analysing the .info() output carefully?"
    return("Success")

def test_1_5_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer).encode('utf8')).hexdigest() == "709a23220f2c3d64d1e1d6d18c4d5280f8d82fca", "Your answer is incorrect. Are you analysing the .describe() output carefully?"
    return("Success")

def test_1_5_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer).encode('utf8')).hexdigest() == "da4b9237bacccdf19c0760cab7aec4a8359010b0", "Your answer is incorrect. Are you analysing the .describe() output carefully?"
    return("Success")

def test_1_5_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer).encode('utf8')).hexdigest() == "b6589fc6ab0dc82cf12099d1c2d40ab994e8410c", "Your answer is incorrect. Are you analysing the .describe() output carefully?"
    return("Success")

def test_1_6_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer).encode('utf8')).hexdigest() == "18bc5956dbf1cc6c9a5baeb624c9e7e472a04c2e", "Your answer is incorrect. Are you gouping and counting correctly?"
    return("Success")

def test_1_6_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer).encode('utf8')).hexdigest() == "18bc5956dbf1cc6c9a5baeb624c9e7e472a04c2e", "Your answer is incorrect. Are you gouping and counting correctly?"
    return("Success")

def test_2_1_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    answer = [x.lower() for x in sorted(answer)]
    assert sha1(str(answer).encode('utf8')).hexdigest() == '3a0c24363ae25561ab1b85b3a8e5cb943f5dd767', "Your answer is incorrect. Are you analysing the dataframe correctly?"
    return("Success")

def test_2_1_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    answer = [x.lower() for x in sorted(answer)]
    assert sha1(str(answer).encode('utf8')).hexdigest() == '379063041b773c8c881c462b975dd1356ab2d3b8', "Your answer is incorrect. Are you analysing the dataframe correctly?"
    return("Success")

def test_2_1_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    answer = [x.lower() for x in sorted(answer)]
    assert sha1(str(answer).encode('utf8')).hexdigest() == '857945940e3729ddb2f76ae4ecc75754a1c7c187', "Your answer is incorrect. Are you analysing the dataframe correctly?"
    return("Success")

def test_2_1_4(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    answer = [x.lower() for x in sorted(answer)]
    assert sha1(str(answer).encode('utf8')).hexdigest() == '97d170e1550eee4afc0af065b78cda302a97674c', "Your answer is incorrect. Are you analysing the dataframe correctly?"
    return("Success")

def test_2_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert 'median' in str(list(answer)[0]), "Make sure your using the simple imputer with the median strategy."
    assert str(list(answer)[1]) == 'StandardScaler()', "Make sure you are using the standard scaler for scaling of the data."
    return("Success")

def test_2_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert 'int' in str(list(answer)[1]), "Make sure you are specifying the data type to be int for the encoding."
    assert 'ignore' in str(list(answer)[1]), "Make sure you are using the 'ignore' method to handle unknown cases."
    assert 'most_frequent' in str(list(answer)[0]), "Make sure your using the simple imputer with the most_frequent strategy."
    return("Success")

def test_2_4(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert 'if_binary' in str(list(answer)[1]), "Make sure you are dropping the binary variables."
    assert 'int' in str(list(answer)[1]), "Make sure you are specifying the data type to be int for the encoding."
    assert 'most_frequent' in str(list(answer)[0]), "Make sure your using the simple imputer with the most_frequent strategy."
    return("Success")

def test_2_5(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert str(list(answer.transformers)).count('pipeline') == 3, "Make sure you are including all three pipelines."
    assert str(list(answer.transformers)).count('most_frequent') == 2, "Make sure you are including all three pipelines."
    assert str(list(answer.transformers)).count('int') == 2, "Make sure you are including all three pipelines."
    return("Success")

def test_3_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.shape == (5, 4), "The dimensions of you solution is incorrect. Are you setting up the model correctly?"
    assert sorted(list(answer.columns)) == ['fit_time', 'score_time', 'test_score', 'train_score'], "Your dataframe contains the incorrect columns. Are you setting up the mocel correctly?"
    assert min(answer['test_score']) > 0 and max(answer['test_score']) < 1, "The range of your test scores is incorrect. Are you fitting the model properly?"
    assert min(answer['train_score']) > 0 and max(answer['train_score']) < 1, "The range of your training scores is incorrect. Are you fitting the model properly?"
    return("Success")

def test_3_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert 'random_state=77' in str(list(answer)), "Make sure you are using a random state of 77."
    assert 'n_estimators=10' in str(list(answer)), "Make sure you are using a setting n_estimators to 10."
    assert str(list(answer)).count('pipeline') == 3, "Make sure you are including the column transformer from earlier."
    assert str(list(answer)).count('most_frequent') == 2, "Make sure you are including the column transformer from earlier."
    assert str(list(answer)).count('int') == 2, "Make sure you are including the column transformer from earlier."
    return("Success")

def test_3_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.shape == (5, 4), "The dimensions of you solution is incorrect. Are you setting up the model correctly?"
    assert sorted(list(answer.columns)) == ['fit_time', 'score_time', 'test_score', 'train_score'], "Your dataframe contains the incorrect columns. Are you setting up the mocel correctly?"
    assert min(answer['test_score']) > 0.5 and max(answer['test_score']) < 1, "The range of your test scores is incorrect. Are you fitting the model properly?"
    assert min(answer['train_score']) > 0.70 and max(answer['train_score']) < 1, "The range of your training scores is incorrect. Are you fitting the model properly?"
    return("Success")

def test_3_5(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1((answer + 'w').encode('utf8')).hexdigest() == "5d58a9df520ee6fc64f92f87a8917efc36f686af", "Your answer is incorrect. How does the training accuracy compare to the test accuracy?"
    return("Success")

def test_3_6(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1((answer + 'k').encode('utf8')).hexdigest() == "54592d9d74c9a90049bdf3693bb2cc3db0334a02", "Your answer is incorrect. Are you comparing the models correctly?"
    return("Success")

def test_3_7(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert str(list(answer.get_params()['estimator'])).count('pipeline') == 3, "Make sure you are including the main pipeline from earlier in the randomized cross validation."
    assert str(list(answer.get_params()['estimator'])).count('most_frequent') == 2, "Make sure you are including the main pipeline from earlier in the randomized cross validation."
    assert str(list(answer.get_params()['estimator'])).count('int') == 2, "Make sure you are including the main pipeline from earlier in the randomized cross validation."
    assert answer.get_params()['cv'] == 5, "Make sure you are using 5-fold cross validation."
    assert 'random_state=77' in str(answer.get_params()['estimator__randomforestclassifier']), "Make sure you are using a random state of 77."
    assert answer.get_params()['n_iter'] == 5, "Make sure you are iterating 5 times."
    assert answer.get_params()['return_train_score'] == True, "Make sure you are returning the training score."
    return("Success")

def test_3_8(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sorted(list(answer.columns)) == ['mean_fit_time','mean_test_score','param_randomforestclassifier__max_depth','rank_test_score'], "Make sure you are selecting the required columns."
    assert min(answer['mean_test_score']) > 0.2 and max(answer['mean_test_score']) < 1, "The range of your test scores is incorrect. Are you selecting the correct column?"
    assert min(answer['rank_test_score']) == 1 and max(answer['rank_test_score']) == 5, "The range of your test scores ranking is incorrect. Are you selecting the correct column?"
    return("Success")

def test_3_9_old(answer1, answer2):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert isinstance(answer1, int), "Your answer should be of type int." 
    assert answer1 in range(10, 100), "Your value for best depth is incorrect. Are you using the .best_param function?"
    assert round(answer2, 2) in np.linspace(0,1,101), "Your value for the best depth score is incorrect. Are you using the .best_score function?"
    return("Success")

def test_3_9(answer, answer1, answer2):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.best_params_['randomforestclassifier__max_depth'] == answer1, "Your value for best depth is incorrect. Are you using the .best_param function?"
    assert answer.best_score_ == answer2, "Your value for the best depth score is incorrect. Are you using the .best_score function?"
    return("Success")

def test_4_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,2)).encode('utf8')).hexdigest() == "3531e3225cc01d9da74d8f4a7ca4b3138d2b7495", "Your answer is incorrect. Are you scoring the model on the training set properly?"
    return("Success")

def test_4_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert round(answer, 2) in np.linspace(0,1,101), "Your answer is incorrect. Are you scoring the model on the test set properly?"
    return("Success")

def test_5_1(answer1,answer2):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer1.shape == (4457, 2), "The dimensions of training set is incorrect. Are you splitting correctly?"
    assert answer2.shape == (1115, 2), "The dimensions of the test set is incorrect. Are you splitting correctly?"
    assert list(answer1.loc[1283]) == ['ham', 'Yes i thought so. Thanks.'], "Make sure you are setting your random state to 123."
    assert list(answer2.loc[1284]) == ['ham', "But if she.s drinkin i'm ok."], "Make sure you are setting your random state to 123"
    return("Success")

def test_5_2(answer1,answer2,answer3,answer4):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer3 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer3 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer1.shape == (4457,) or answer1.shape == (4457, 1), "The dimensions of the training set is incorrect. Are you splitting correctly?"
    assert answer2.shape == (1115,) or answer2.shape == (1115, 1), "The dimensions of the test set is incorrect. Are you splitting correctly"
    assert answer3.shape == (4457,), "The dimensions of the training set is incorrect. Are you splitting correctly? Are you using single brackets?"
    assert answer4.shape == (1115,), "The dimensions of the test set is incorrect. Are you splitting correctly? Are you using single brackets?"
    return("Success")

def test_5_4(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,3)).encode('utf8')).hexdigest() == "01851289a0bf981b02d6d1fc97cea48cd4eaa597", "The average text length is incorrect. Are you taking the mean?"
    return("Success")

def test_5_5(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1((answer + 'm').encode('utf8')).hexdigest() == "799a12625051a5fe5760d613b2ccbb929f82169b", "Your answer is incorrect. Are there any categories present?"
    return("Success")

def test_5_6():
    assert 'sklearn' in sys.modules, "Make sure you are importing 'GridSearchCV' and  'RandomizedSearchCV' from the sklearn module."
    return("Success")

def test_5_7(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.shape == (4457, 7682), "The dimensions of transformed dataset is incorrect. Are you using the transform fucntion?"
    assert str(answer.dtype) == 'int64', "The data type of transformed dataset is incorrect. Are you using the count vectorizer function?"
    return("Success")

def test_5_8(answer):
    assert sha1(str(answer).encode('utf8')).hexdigest() == "0850971bce4fe433dcd19ff9297591dc93750712", "Your solution is incorrect. Please try again."
    return("Success")

def test_5_10(answer1,answer2):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert 'binary=True' in str(list(answer1)), "Make sure you are setting binary to true in your pipeline."
    assert 'most_frequent' in str(list(answer1)), "Make sure you using the most frequent strategy for the Dummy Classifier."
    assert answer2.shape == (5, 4), "The dimensions of you solution is incorrect. Are you using 5-fold cross validation?"
    assert sorted(list(answer2.columns)) == ['fit_time', 'score_time', 'test_score', 'train_score'], "Your dataframe contains the incorrect columns. Are you setting up the mocel correctly?"
    assert min(answer2['test_score']) > 0.80 and max(answer2['test_score']) < 1, "The range of your test scores is incorrect. Are you fitting the model properly?"
    assert min(answer2['train_score']) > 0.80 and max(answer2['train_score']) < 1, "The range of your training scores is incorrect. Are you fitting the model properly?"
    return("Success")

def test_5_11(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    answer = [round(x,3) for x in list(answer)]
    sha1(str(answer[2]).encode('utf8')).hexdigest() == "b5f478e5b18e752119bbb379d963c2357c5da422", "Your answer for the training score is incorrect. Are you taking the mean?"
    sha1(str(answer[3]).encode('utf8')).hexdigest() == "b5f478e5b18e752119bbb379d963c2357c5da422", "Your answer for the test incorrect. Are you taking the mean?"
    return("Success")

def test_5_12(answer):
    assert answer.shape == (5, 4), "The dimensions of you solution is incorrect. Are you using 10-fold cross validation?"
    assert sorted(list(answer.columns)) == ['fit_time', 'score_time', 'test_score', 'train_score'], "Your dataframe contains the incorrect columns. Are you setting up the mocel correctly?"
    assert min(answer['test_score']) > 0.96 and max(answer['test_score']) < 1, "The range of your test scores is incorrect. Are you fitting the model properly?"
    assert min(answer['train_score']) > 0.99 and max(answer['train_score']) < 1, "The range of your training scores is incorrect. Are you fitting the model properly?"
    return("Success")

def test_5_13(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    answer = [round(x,3) for x in list(answer)]
    sha1(str(answer[2]).encode('utf8')).hexdigest() == "e6da655eed7b9306b624c65d43d42651284d84fd", "Your answer for the training score is incorrect. Are you taking the mean?"
    sha1(str(answer[3]).encode('utf8')).hexdigest() == "84a16d4bec128d66be2277661080f21624dcb546", "Your answer for the test incorrect. Are you taking the mean?"
    return("Success")

def test_5_14(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1((answer + 'p').encode('utf8')).hexdigest() == "b920fb3adf34d73af2cb7b5c93d8efbc94cd36f7", "Your answer is incorrect. Are you comparing both classifiers correctly?"
    return("Success")














