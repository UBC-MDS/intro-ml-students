from hashlib import sha1
import pandas as pd
import pytest
import altair
import sys


def test_1_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer.lower() + '9').encode('utf8')).hexdigest(
    ) == '68ee74f7d6afe0164fe0f1197aa9177c946d8834', "Your answer is incorrect. Please try again."
    return "Success"


def test_1_2(answer1,answer2):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    answer1 = sorted([x.lower() for x in answer1])
    answer2 = sorted([x.lower() for x in answer2])
    assert sha1(str(answer1).encode('utf8')).hexdigest() == '88469548e6757e027666deef9d9979d33c6aec00', "Your answer for 'supervised' is incorrect. Please try again."
    assert sha1(str(answer2).encode('utf8')).hexdigest() == 'aaf3f256bedabb9adfec9331a0ac36955c9aba35', "Your answer for 'unsupervised' is incorrect. Please try again."
    return "Success"

def test_1_3_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer.lower()).encode('utf8')).hexdigest(
    ) == 'ea585073ff8843ad70c48735f623e8440e73cd57', "Your answer for 'ii' in incorrect. Please try again"
    return("Success")

def test_1_3_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer.lower() + 'k').encode('utf8')).hexdigest(
    ) == '5db2ee1c744c0068a109f910640164eeab637e08', "Your answer for 'iii' in incorrect. Please try again"
    return("Success")

def test_2_1_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(answer.strip('s').lower().encode('utf8')).hexdigest(
    ) == 'c3499c2729730a7f807efb8676a92dcb6f8a3f8f', 'The answer for rows is incorrect. Please try again.'
    return("Success")

def test_2_1_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(answer.strip('s').lower().encode('utf8')).hexdigest(
    ) == '4b7615dce52c4c05ce4e1d374e9c61a13717ac7c', 'The answer for inputs is incorrect. Please try again.'
    return("Success")

def test_2_1_4(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    temp = sha1(answer.lower().encode('utf8')).hexdigest()
    assert temp == 'ea9ffb6eb4a5f167f6a29e1140b39165d47734fd' or temp == 'a5a58e1868cd78e185dfe4920fc5955ef0c3f9a9', 'The answer for training is incorrect. Please try again.'
    return("Success")

def test_2_2a(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer + 12).encode('utf8')).hexdigest(
    ) == 'fa35e192121eabf3dabf9f5ea6abdbcbc107ac3b', "Your answer is incorrect. Keep in mind that the target is not part of the features."
    return "Success"


def test_2_2b(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(sorted(answer)).encode('utf8')).hexdigest(
    ) == '5ab0f64f0f9c950d61582cc9ff51210b87551599', "The names of features are incorrect. Besides the target column, what other columns are in the dataframe?"
    return "Success"


def test_2_2c(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer + 11).encode('utf8')).hexdigest(
    ) == "fa35e192121eabf3dabf9f5ea6abdbcbc107ac3b", "The number of classes is incorrect. Think about the possible values that the target can take."
    return("Success")


def test_3_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer).encode('utf8')).hexdigest(
    ) == "fe5dbbcea5ce7e2988b8c69bcfdfde8904aabc1f", "The number of features is incorrect. The 'len' and '.columns' functions may be useful here."
    return "Success"

def test_3_3(answer1, answer2):
    assert not answer1 is None, "The 'X' variable does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "The 'y' variable does not exist. Have you passed in the correct variable?"
    assert answer1.shape == (100, 8), "The size of 'X' is incorrect. Are you dropping the target column?"
    assert answer2.shape == (100,), "The size of 'y' is incorrect. Are you only selecting the target column?"
    assert 'diagnosis' not in answer1.columns, "Make sure you are not selecting the target column as part of 'X'"
    return "Success"


def test_4_1():
    assert 'sklearn' in sys.modules, "Make sure you are importing the 'DummyClassifier' function from the 'sklearn.dummy' module."
    return("Success")


def test_4_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.strategy == "stratified", "Make sure you are creating a stratified dummy classifier"
    assert answer.random_state == 1, "Make sure you are setting a random state of 1"
    return "Success"


def test_4_3(answer1,answer2):
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert list(answer1.classes_) == ['N', 'O'], "Your model contains the incorrect classes. Are you calling the 'fit' function and passing 'X' and 'y'?"
    assert list(answer2).count('N') == 87, "Predictions are incorrect. Are you fitting the model and predicting correctly?"
    assert list(answer2).count('O') == 13, "Predictions are incorrect. Are you fitting the model and predicting correctly?"
    return "Success"


def test_4_5(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert round(answer[0],2) == 0.77, "The accuracy value for 'stratified' is incorrect."
    assert round(answer[1],2) == 0.88, "The accuracy value for 'most_frequent' is incorrect."
    assert round(answer[2],2) == 0.45, "The accuracy value for 'uniform' is incorrect."
    return "Success"


def test_4_6(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.shape == (3, 2), "The dimensions of the dataframe is incorrect. Are you creating it properly?"
    assert sorted(list(answer.columns)) == ['accuracy', 'strategy'], "The columns names are incorrect. Please refer to the question."
    assert list(answer.loc[0].values) == ['stratified', 0.77], "Your dataframe values are incorrect."
    assert list(answer.loc[1].values) == ['most_frequent', 0.88], "Your dataframe values are incorrect."
    assert list(answer.loc[2].values) == ['uniform', 0.45], "Your dataframe values are incorrect."
    return "Success"


def test_4_7(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    try:
        answer.mark.type == "bar", "Your plot is not a bar plot. Make sure you are using the 'mark_bar()' function"
    except AttributeError as error:
        assert answer.mark == "bar", "Your plot is not a bar plot. Make sure you are using the 'mark_bar()' function"
    assert 'strategy' in str(answer.encoding.x.shorthand) or 'strategy' in str(answer.encoding.x.field), "Make sure you are plotting the `strategy` variable on the x-axis."
    assert 'accuracy' in str(answer.encoding.y.shorthand) or 'accuracy' in str(answer.encoding.y.field), "Make sure you are plotting the `accuracy` variable. on the y-axix"
    assert not answer.title == 'Undefined', "Make sure you are providing a title for the plot."
    return "Success"


def test_4_8(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    case1 = (sha1(str(answer.lower() + 'k').encode('utf8')).hexdigest(
    ) == 'b86f6db284d374188d561d46b45b188d5631609a')
    case2 = (sha1(str(answer.lower()).encode('utf8')).hexdigest() == '7103f0aa18aa01027dc1e1e8783aeaa8b6a54feb')
    assert (case1 or case2) == True, 'Your answer is incorrect. Please try again'
    
    return "Success"


def test_5_1(answer1, answer2):
    assert not answer1 is None, "The 'X' variable does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "The 'y' variable does not exist. Have you passed in the correct variable?"
    assert answer1.shape == (414, 5), "The size of 'X' is incorrect. Are you dropping the price column?"
    assert answer2.shape == (414,), "The size of 'y' is incorrect. Are you only selecting the target column?"
    assert 'price' not in answer1.columns, "Make sure you are not selecting the price column as part of 'X'"
    return "Success"


def test_5_2():
    assert 'sklearn' in sys.modules, "Make sure you are importing the 'DummyRegressor' function from the 'sklearn.dummy' module."
    return("Success")


def test_5_3(answer1,answer2):
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer1.get_params()['strategy'] == 'mean', "Make sure you are creating a 'mean' dummy classifier."
    assert round(list(set(answer2))[0],2) == 37.98, "Predictions are incorrect. Are you fitting the model and predicting correctly?"
    return "Success"


def test_5_4(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer == 0.0, "We expect your answer to be 0.0. Are you doing fitting the model correctly?"
    return "Success"


def test_5_5(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer.upper() + 'k').encode('utf8')).hexdigest(
    ) == 'a1382f8f61bcba9266e29a3fce9ccd7c9b961459', 'Your answer is incorrect. Please try again'
    return("Success")
