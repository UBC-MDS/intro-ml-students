from hashlib import sha1
import pandas as pd
import pytest
import altair
import sys


def test_1_1():
    assert 'sklearn' in sys.modules, "Make sure you are importing 'train_test_split' from the sklearn module."
    return("Success")

def test_1_2(answer1,answer2):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer1.shape == (81, 6), "The dimensions of training set is incorrect. Are you splitting correctly?"
    assert answer2.shape == (21, 6), "The dimensions of the test set is incorrect. Are you splitting correctly?"
    assert list(answer1.loc[50]) == [0.0, 5.0, 1.0, 0.0, 3.0, 0.0], "Make sure you are setting your random state to 77."
    assert list(answer2.loc[5]) == [0.0, 7.0, 1.0, 1.0, 4.0, 0.0], "Make sure you are setting your random state to 77"
    return("Success")

def test_1_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.shape == (8,6), "The dimensions of your solution is incorrect. Are you using the describe function?"
    assert 'std' in list(answer.index), "Your solution is missing some values. Are you using the describe function?"
    assert 'PhoneReach' in list(answer.columns), "Your solution is missing some columns. Are you using the correct dataframe?"
    assert list(answer.iloc[7]) == [1.0, 10.0, 1.0, 1.0, 5.0, 1.0], "Your solution is missing some values. Are you using the describe function?"
    return("Success")

def test_1_5(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,2)).encode('utf8')).hexdigest() == "8ecc5701b7c3d81ab5ad490cebb860b82dc069c2", "Your solution is incorrect. The 'mean' function may be useful here."
    return("Success")

def test_2_1(answer1,answer2,answer3,answer4):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer3 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer3 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer1.shape == (81, 5), "The dimensions of the training set is incorrect. Are you splitting correctly?"
    assert answer2.shape == (21, 5), "The dimensions of the test set is incorrect. Are you splitting correctly"
    assert answer3.shape ==  (81,), "The dimensions of the training set is incorrect. Are you splitting correctly?"
    assert answer4.shape == (21,), "The dimensions of the test set is incorrect. Are you splitting correctly"
    assert 'Breakfast' not in list(answer1.columns), "Make sure the target variable is not part of your X dataset."
    assert list(answer1.loc[50]) == [0.0, 5.0, 1.0, 0.0, 3.0], "Make sure you are using random state 77."
    assert list(answer2.loc[5]) == [0.0, 7.0, 1.0, 1.0, 4.0], "Make sure you are using random state 77."
    return("Success")

def test_2_2(answer1,answer2):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer1,2)).encode('utf8')).hexdigest() == "8ecc5701b7c3d81ab5ad490cebb860b82dc069c2", "Your answer is incorrect. Are you setting up the model correctly?"
    assert sha1(str(round(answer2,2)).encode('utf8')).hexdigest() == "6784503f48bed278f87e4f1ff9eb9278d4065759", "Your answer is incorrect. Are you setting up the model correctly?"
    return("Success")

def test_2_3(answer1,answer2,answer3):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer3.random_state == 77, "Make sure you are setting your model's random state to 77"
    assert sha1(str(round(answer1,2)).encode('utf8')).hexdigest() == "15e4f66808acb4ab8ac72b8c56eba22310069e7f", "Your answer is incorrect. Are you setting up the model correctly?"
    assert sha1(str(round(answer2,2)).encode('utf8')).hexdigest() == "b5adc604ee2bac453cd02eafdad0df69f73001d2", "Your answer is incorrect. Are you setting up the model correctly?"
    return("Success")

def test_2_5(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer.lower() + 'p').encode('utf8')).hexdigest() == "ac78b022715c5b8357b4dca8045e8463b4de2124", "Your solution is inocrrect. Are you examinng the test scores properly?"
    return("Success")

def test_2_6(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer.lower() + 'w').encode('utf8')).hexdigest() == "8a85c06c4eba8aba733a4be991bd4b4c6b4e4581", "Your solution is inocrrect. How does the training and test scores compare for both models?"
    return("Success")

def test_2_7(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer.lower() + 'a').encode('utf8')).hexdigest() == "e0c9035898dd52fc65c41454cec9c4d2611bfb37", "Your solution is inocrrect. Think about how the dummy classifier is making its predictions?"
    return("Success")

def test_3_1():
    assert 'sklearn' in sys.modules, "Make sure you are importing 'cross_validate' from the sklearn module."
    return("Success")

def test_3_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.random_state == 77, "Make sure you are setting your model's random state to 77"
    return("Success")

def test_3_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sorted(list(answer.keys())) == ['fit_time', 'score_time', 'test_score', 'train_score'], "Make sure you are recording the times as well as scores for training and testing"
    assert sum([len(x) for x in answer.values()]) == 20, "Make sure you are uisng 5-fold cross validation"
    assert min(answer['test_score']) > 0 and max(answer['test_score']) < 1, "The range of your test scores is incorrect. Are you fitting the model properly?"
    assert min(answer['train_score']) > 0 and max(answer['train_score']) < 1, "The range of your training scores is incorrect. Are you fitting the model properly?"
    return("Success")

def test_3_4(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.shape == (5, 4), "The dimensions of your dataframe is incorrect. Are you tranforming the cv socres dictionary?"
    assert sorted(list(answer.columns)) == ['fit_time', 'score_time', 'test_score', 'train_score'], "Your dataframe contains incorrect columns. Are you transforming the cv scores dictionary?"
    return("Success")

def test_3_5(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    answer = list(answer)
    assert min(answer) > 0 and max(answer) < 1, "The range of your values are incorrect. Are you taking the mean?"
    return("Success")

def test_4_1(answer1,answer2,answer3,answer4):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer3 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer3 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer1.shape == (81, 5), "The dimensions of the training set is incorrect. Are you splitting correctly?"
    assert answer2.shape == (21, 5), "The dimensions of the test set is incorrect. Are you splitting correctly"
    assert answer3.shape ==  (81,), "The dimensions of the training set is incorrect. Are you splitting correctly?"
    assert answer4.shape == (21,), "The dimensions of the test set is incorrect. Are you splitting correctly"
    assert list(answer1.loc[50]) == [0.0, 5.0, 1.0, 0.0, 3.0], "Make sure you are using random state 77."
    assert list(answer2.loc[5]) == [0.0, 7.0, 1.0, 1.0, 4.0], "Make sure you are using random state 77."
    return("Success")

def test_4_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sum([len(x) for x in answer.values()]) == 75, "Make sure you are iterating 25 times"
    assert min(answer['mean_train_score']) > 0 and max(answer['mean_train_score']) < 1, "The range of your mean training scores is incorrect. Are you fitting the model properly?"
    assert min(answer['mean_cv_score']) > 0 and max(answer['mean_cv_score']) < 1, "The range of your test scores is incorrect. Are you fitting the model properly?"
    return("Success")

def test_4_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.shape == (25, 3), "The dimensions of your dataframe is incorrect. Are you converting the results dictionary to a dataframe?"
    assert sorted(list(answer.columns)) == ['mean_cv_score', 'mean_train_score', 'min_samples_split'], "Your datafame contains incorrect columns. Are you converting the results dictionary to a dataframe?"
    return("Success")

def test_4_4(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.shape == (50, 3), "The dimensions of your dataframe is incorrect. Are you melting the dataframe properly?"
    assert sorted(list(answer.columns)) == ['accuracy', 'min_samples_split', 'score_type'], "Your datafame contains incorrect columns. Are you melting on the correct variables?"
    assert set(answer['score_type']) == {'mean_cv_score', 'mean_train_score'}, "Your dataframe contains incorrect values. Are you melting on the correct variables?"
    return("Success")

def test_4_5(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    try:
        answer.mark.type == "line", "Your plot is not a line plot. Make sure you are using the 'mark_line()' function"
    except AttributeError as error:
        assert answer.mark == "line", "Your plot is not a line plot. Make sure you are using the 'mark_line()' function"
    assert 'min_samples_split' in str(answer.encoding.x.shorthand) or 'min_samples_split' in str(answer.encoding.x.field), "Make sure you are plotting the `strategy` variable on the x-axis."
    assert 'accuracy' in str(answer.encoding.y.shorthand) or 'accuracy' in str(answer.encoding.y.field), "Make sure you are plotting the `accuracy` variable. on the y-axix"
    assert not answer.title == 'Undefined', "Make sure you are providing a title for the plot."
    return("Success")

def test_4_6(answer):
    assert sha1(str(float(answer)).encode('utf8')).hexdigest() == "2493779251de822754e7d9cbd06e551dfa7fcd2b", "Your answer is incorrect. Are you finding the max value amongst all splits?"
    return("Success")

def test_4_7(answer):
    answer = answer.get_params()['min_samples_split']
    assert sha1(str(answer).encode('utf8')).hexdigest() == "0a57cb53ba59c46fc4b692527a38a87c78d84028", "Make sure you are using the value for the best split in your model."
    return("Success")

def test_4_8(answer):
    assert sha1(str(round(answer,2)).encode('utf8')).hexdigest() == "7fa45d6b92de242183d0c4b5ee972ec569ea25c2", "Your answer is incorrect. Are you fitting and scoring the model properly?"
    return("Success")

def test_4_9(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer.lower() + 'c').encode('utf8')).hexdigest() == "bdb480de655aa6ec75ca058c849c4faf3c0f75b1", "Your solution is inocrrect. Are you comparing the scores closely?"
    return("Success")

def test_4_10(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer.lower() + 'z').encode('utf8')).hexdigest() == "57f378cca8e1bd5ea94400ff922e6451409e0765", "Your solution is inocrrect. Are there multiple reasons?"
    return("Success")
