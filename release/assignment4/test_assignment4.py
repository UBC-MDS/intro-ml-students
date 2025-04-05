from hashlib import sha1
import pandas as pd
import pytest
import altair
import sys


def test_1_1(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer + 9).encode('utf8')).hexdigest() == "0ade7c2cf97f75d009975f4d720d1fa6c19f4897", "Your answer is incorrect. Are you counting correctly?"
    return("Success")

def test_1_2(answer1,answer2):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer1.shape == (5197, 13), "The dimensions of training set is incorrect. Are you splitting correctly?"
    assert answer2.shape == (1300, 13), "The dimensions of the test set is incorrect. Are you splitting correctly?"
    assert list(answer1.loc[2415])[0:4] == [8.4, 0.18, 0.42, 5.1], "Make sure you are setting your random state to 2020."
    assert list(answer2.loc[2158])[0:4] == [7.2, 0.34, 0.44, 4.2], "Make sure you are setting your random state to 2020"
    return("Success")

def test_1_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer + 10).encode('utf8')).hexdigest() == "12c6fc06c99a462375eeb3f43dfd832b08ca9e17", "The dimension is incorrect. Are you removing the target column?"
    return("Success")

def test_1_4(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.shape == (8, 12), "The dimensions of your solution is incorrect. Are you using the describe function?"
    assert 'std' in list(answer.index), "Your solution is missing some values. Are you using the describe function?"
    assert 'fixed_acidity' in list(answer.columns), "Your solution is missing some columns. Are you using the correct dataframe?"
    assert list(answer.iloc[7]) == [15.9, 1.58, 1.66, 65.8, 0.611, 289.0, 440.0, 1.03898, 4.01, 2.0, 14.9, 9.0], "Your solution is missing some values. Are you using the describe function?"
    return("Success")

def test_1_5(answer1,answer2):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer1,2)).encode('utf8')).hexdigest() == "09dcc1ab6de1c8860ed2dc6936960960658ac4ee", "The average red pH is incorrect. The groupby function may be useful here."
    assert sha1(str(round(answer2,2)).encode('utf8')).hexdigest() == "0bacfaf4756f1210811f3e4496c6fe1cdb0828fb", "The average white pH is incorrect. The groupby function may be useful here."
    return("Success")

def test_1_6(answer1,answer2):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer1,2)).encode('utf8')).hexdigest() == "f1719af4227f44867abd76e5be145b526d6a2b92", "The average red alcohol content is incorrect. The groupby function may be useful here."
    assert sha1(str(round(answer2,2)).encode('utf8')).hexdigest() == "ffbd1b866d1ea6bf16cfec9305857fb3e1e03938", "The average white white alcohol content is incorrect. The groupby function may be useful here."
    return("Success")

def test_1_7(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    try:
        answer.mark.type == "bar", "Your plot is not a bar plot. Make sure you are using the 'mark_bar()' function"
    except AttributeError as error:
        assert answer.mark == "bar", "Your plot is not a bar plot. Make sure you are using the 'mark_bar()' function"
    assert answer.encoding.x.shorthand == 'style:N' or answer.encoding.x.field == 'style:N' or answer.encoding.x.shorthand == 'style' or answer.encoding.x.field == 'style', "Make sure you are plotting 'style' on the x-axis."
    assert answer.encoding.y.shorthand == 'count():Q' or answer.encoding.y.aggregate == 'count()' or answer.encoding.y.shorthand == 'count()' or answer.encoding.y.aggregate == 'count', "Make sure you are plotting 'count()' on the y-axis."
    return ("Success")


def test_2_1(answer1,answer2,answer3,answer4):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer3 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer3 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer1.shape == (5197, 12), "The dimensions of the training set is incorrect. Are you splitting correctly?"
    assert answer2.shape == (1300, 12), "The dimensions of the test set is incorrect. Are you splitting correctly"
    assert answer3.shape ==  (5197,), "The dimensions of the training set is incorrect. Are you splitting correctly?"
    assert answer4.shape == (1300,), "The dimensions of the test set is incorrect. Are you splitting correctly"
    assert 'style' not in list(answer1.columns), "Make sure the target variable is not part of your X dataset."
    return("Success")

def test_2_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.shape == (5197, 5197), "The dimensions of your solution is incorrect. Are you computing all pairwise distances?"
    assert  list(answer[0][0:5]) == [0.0,119.48948388848954,156.7993125354908,110.78584353409059,46.52215770372227], "Some of your values are incorrect. Are you computing all pairwise distances?"
    assert  list(answer[1][0:5]) == [119.48948388848954,0.0,38.11165347035107,28.5481687246243,82.09932764365128], "Some of your values are incorrect. Are you computing all pairwise distances?"
    return("Success")

def test_2_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer).encode('utf8')).hexdigest() == "74c92134b13c3114b5d973512d082cd73722f969", "Your answer is incorrect. Are you using the fill_diagonal function?"
    return("Success")

def test_2_4(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,2)).encode('utf8')).hexdigest() == "5cca59003c7415ffd48677288583be39d9b5aee0", "Your answer is incorrect. Are you using the fill_diagonal function?"
    return("Success")

def test_2_5(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer).encode('utf8')).hexdigest() == "fc2dcda259b73344ef93a7517f91c126741c4065", "Your solution is incorrect. The np.argmin() might be useful here."
    return("Success")

def test_2_6(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,2)).encode('utf8')).hexdigest() == "04c95c8c7c421bb9110f33c1c3da45df9d48e868", "Your solution is incorrect. The np.argmin() might be useful here."
    return("Success")

def test_3_1(answer1,answer2,answer3):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer3 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer3.strategy == 'most_frequent', "Make sure your model is using the most frequent strategy."
    assert sha1(str(round(answer1,2)).encode('utf8')).hexdigest() == "0cf1aeac0372e10da230a32e74c9b6e68dca7342", "Your training score is incorrect. Are you building the model correctly?"
    assert sha1(str(round(answer2,2)).encode('utf8')).hexdigest() == "b479a13e911c838f0b4e10ba449f6fad64243a55", "Your test score is incorrect. Are you building the model correctly?"
    return("Success")

def test_3_2(answer1,answer2,answer3):
    assert not answer1 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert not answer3 is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer3.n_neighbors == 1, "Make sure you are setting n_neighbor to 1."
    assert sha1(str(round(answer1,2)).encode('utf8')).hexdigest() == "e8dc057d3346e56aed7cf252185dbe1fa6454411", "Your training score is incorrect. Are you building the model correctly?"
    assert sha1(str(round(answer2,2)).encode('utf8')).hexdigest() == "33493e5f53a2a235a6fe783b9b97f25c83c56178", "Your validation score is incorrect. Are you building the model correctly?"
    return("Success")

def test_3_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer + 'w').encode('utf8')).hexdigest() == "a13c41da4efbb5bf6cd3cd05fffa5c5b182b8727", "Your solution is incorrect. Please try again."
    return("Success")

def test_3_4(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer + 'k').encode('utf8')).hexdigest() == "a1382f8f61bcba9266e29a3fce9ccd7c9b961459", "Your solution is incorrect. Please try again"
    return("Success")

def test_3_5(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer + 'c').encode('utf8')).hexdigest() == "680323f027883c78b36a7d555bd184bc816cb4ea", "Your solution is incorrect. Please try again"
    return("Success")

def test_3_6(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer).encode('utf8')).hexdigest() == "88b33e4e12f75ac8bf792aebde41f1a090f3a612", "Your answer is incorrect. Please try again."
    return("Success")

def test_3_7(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer + 'x').encode('utf8')).hexdigest() == "078a28d88163d1e73a6b3dc4659d94db6be4b55f", "Your answer is incorrect. Pleas try again."
    return("Success")

def test_3_8(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sorted(list(answer.keys())) == ['mean_cv_score', 'mean_train_score', 'n_neighbors'], "Make sure you are recording the times as well as scores for training and cross validation"
    assert sum([len(x) for x in answer.values()]) == 27, "Make sure you are iterating over n_neighbors values over every second number from 2 to 20 (inclusive)"
    assert min(answer['mean_cv_score']) > 0 and max(answer['mean_cv_score']) < 1, "The range of your cross validation scores is incorrect. Are you fitting the model properly?"
    assert min(answer['mean_train_score']) > 0 and max(answer['mean_train_score']) < 1, "The range of your training scores is incorrect. Are you fitting the model properly?"
    return("Success")

def test_3_9(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.shape == (18, 3), "The dimensions of your dataframe is incorrect. Are you tranforming the cv socres dictionary?"
    assert sorted(list(answer.columns)) == ['accuracy', 'n_neighbors', 'score_type'], "Your dataframe contains incorrect columns. Are you transforming the cv scores dictionary?"
    return("Success")

def test_3_10(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    try:
        answer.mark.type == "line", "Your plot is not a bar plot. Make sure you are using the 'mark_line()' function"
    except AttributeError as error:
        assert answer.mark == "line", "Your plot is not a bar plot. Make sure you are using the 'mark_line()' function"
    # answer.encoding.x.shorthand may return instance of alt.utils.schemapi.UndefinedType which is not iterable. Casting as str prevents TypeError
    assert 'n_neighbors' in str(answer.encoding.x.shorthand)  or  'n_neighbors' in str(answer.encoding.x.field) , "Make sure you are plotting 'n_neighbors' on the x-axis."
    assert 'accuracy' in str(answer.encoding.y.shorthand)  or  'accuracy' in str(answer.encoding.y.field), "Make sure you are plotting 'accuracy' on the y-axis."
    assert not answer.title == 'Undefined', "Make sure you are providing a title for the plot."
    return("Success")

def test_3_11(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer)).encode('utf8')).hexdigest() == "b1d5781111d84f7b3fe45a0852e59758cd7a87e5", "Your solution is incorrect. Please try again. The '.idmax' may be helpful here."
    return("Success")

def test_3_12(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,2)).encode('utf8')).hexdigest() == "0cde346137a4bcd6bbfd76ddf353ed48b7a60bf8", "Your test score is incorrect. Are you fitting the model properly?" 
    return("Success")

def test_4_1():
    assert 'sklearn' in sys.modules, "Make sure you are importing 'SVC' from the sklearn module."
    return("Success")

def test_4_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sorted(list(answer.keys())) == ['gamma', 'mean_cv_score', 'mean_train_score'], "Make sure you are recording gamma as well as scores for training and cross validation"
    assert sum([len(x) for x in answer.values()]) == 12, "Make sure you are iterating over gamma values from 0.1 to 100.0 (inclusive)"
    assert min(answer['mean_cv_score']) > 0 and max(answer['mean_cv_score']) < 1, "The range of your cross validation scores is incorrect. Are you fitting the model properly?"
    assert min(answer['mean_train_score']) > 0 and max(answer['mean_train_score']) < 1, "The range of your training scores is incorrect. Are you fitting the model properly?"
    return("Success")

def test_4_3(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer, 1)).encode('utf8')).hexdigest() == "180505679cfe0cca79bae51fdda0296b7cd9c493", "Your answer is incorrect. Are you examining the results dictionary correctly? The idxmax() function might be useful here."
    return("Success")

def test_4_4(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sorted(list(answer.keys())) == ['C', 'mean_cv_score', 'mean_train_score'], "Make sure you are recording C as well as scores for training and cross validation"
    assert sum([len(x) for x in answer.values()]) == 12, "Make sure you are iterating over gamma values from 0.1 to 100.0 (inclusive)"
    assert min(answer['mean_cv_score']) > 0 and max(answer['mean_cv_score']) < 1, "The range of your cross validation scores is incorrect. Are you fitting the model properly?"
    assert min(answer['mean_train_score']) > 0 and max(answer['mean_train_score']) < 1, "The range of your training scores is incorrect. Are you fitting the model properly?"
    return("Success")

def test_4_5(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer, 1)).encode('utf8')).hexdigest() == "6f50807584714e6e22f44060c99304e24165f7a5", "Your answer is incorrect. Are you examining the results dictionary correctly."
    return("Success")

def test_4_6(answer):
    assert sha1(str(answer + 'u').encode('utf8')).hexdigest() == "bbb8328c16f85703a882da420cc50f1df34c4868", "Your answer is incorrect. Could both gamma and C interact?"
    return("Success")

def test_4_7(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sorted(list(answer.keys())) == ['C', 'gamma', 'train_accuracy', 'valid_accuracy'], "Make sure you are recording C and gamma as well as scores for training and cross validation"
    assert sum([len(x) for x in answer.values()]) == 64, "Make sure you are iterating over gamma and C values simultaneously, both from 0.1 to 100.0 (inclusive)"
    assert min(answer['train_accuracy']) > 0 and max(answer['train_accuracy']) < 1, "The range of your training scores is incorrect. Are you fitting the model properly?"
    assert min(answer['valid_accuracy']) > 0 and max(answer['valid_accuracy']) < 1, "The range of your cross validation scores is incorrect. Are you fitting the model properly?"
    return("Success")


def test_4_8(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.shape == (16, 4), "The dimensions of your dataframe is incorrect. Are you tranforming the cv scores dictionary?"
    assert sum(sorted(answer['valid_accuracy'],reverse=True) == answer['valid_accuracy']) == 16, "Make sure you are sorting based on 'valid_accuracy' in descending order."
    assert sorted(list(answer.columns)) == ['C', 'gamma', 'train_accuracy', 'valid_accuracy'], "Your dataframe contains incorrect columns. Are you transforming the cv scores dictionary?"
    return("Success")

def test_4_9(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(round(answer,2)).encode('utf8')).hexdigest() == "33493e5f53a2a235a6fe783b9b97f25c83c56178", "Your answer is incorrect. Are you computing the best values correctly and using those to fit the model?"
    return("Success")

def test_4_10(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert sha1(str(answer + 'm').encode('utf8')).hexdigest() == "4cd7920fdd3598ce4b7635b6879d03af8ca0fdb1", "Your solution is incorrect. Are you comparing the results correctly?"
    return("Success")


































