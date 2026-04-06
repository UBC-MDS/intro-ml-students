from hashlib import sha1
import pandas as pd
import pytest
import altair
import inspect
import graphviz


def test_1_1_1(answer):
    assert sha1(str(answer.lower()).encode('utf8')).hexdigest() == "dc76e9f0c0006e8f919e0c515c66dbba3982f785", "Your answer is incorrect. Please try again"
    return("Success")

def test_1_1_2(answer):
    assert sha1(str(answer.lower()).encode('utf8')).hexdigest() == "10d735e581f1e2505cd69675691925490e447c44", "Your answer is incorrect. Please try again"
    return("Success")

def test_1_1_4(answer):
    assert sha1(str(answer.lower()).encode('utf8')).hexdigest() == "f8e966d1e207d02c44511a58dccff2f5429e9a3b", "Your answer is incorrect. Please try again"
    return("Success")

def test_1_2(answer):
    assert sha1(str(answer.lower()).encode('utf8')).hexdigest() == "e06b95860a6082ae37ef08874f8eb5fade2549da", "Your answer is incorrect. Please try again"
    return("Success")

def test_1_3(answer):
    assert sha1(str(answer + 9).encode('utf8')).hexdigest() == "bd307a3ec329e10a2cff8fb87480823da114f8f4", "Your answer is incorrect. Are you counting the levels properly?"
    return("Success")

def test_2_2(answer1, answer2):
    assert not answer1 is None, "The 'X' variable does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "The 'y' variable does not exist. Have you passed in the correct variable?"
    assert answer1.shape == (10, 3), "The size of 'X' is incorrect. Are you dropping the target column?"
    assert answer2.shape == (10,), "The size of 'y' is incorrect. Are you only selecting the target column?"
    assert 'target' not in answer1.columns, "Make sure you are not selecting the target column as part of 'X'"
    return("Success")

def test_2_3(answer):
    assert answer.get_params()['criterion'] == 'gini', "Are you initializing a decision tree classifier properly?"
    assert answer.get_params()['splitter'] == 'best', "Are you initializing a decision tree classifier properly?"
    return("Success")

def test_2_4(answer):
    assert isinstance(answer, graphviz.files.Source), "Make sure you are creating a graphviz object of the columns and the decision tree."
    return("Success")

def test_2_5(answer):
    assert sha1(str(answer + 1).encode('utf8')).hexdigest() == "936931368287d72a5bda62a8a3e0d2ed6638fa8f", "The score is incorrect. Are you fitting the model correctly?"
    return("Success")

def test_2_6(answer):
    assert 'predicted' in answer.columns, "Make sure you are naming the columns with the predictions as 'predicted'."
    assert list(answer['predicted']).count('happy') == 6, "Some predicted values are incorrect. Are you fitting the model and predicting correctly?"
    assert list(answer['predicted']).count('unhappy') == 4, "Some predicted values are incorrect. Are you fitting the model and predicting correctly?" 
    return("Success")

def test_2_7(answer):
    assert sha1((answer.lower() + 'k').encode('utf8')).hexdigest() == "c8c2ca9fed1a7c345d7dbc1b7d985364870c874a", "Your answer is incorrect. "
    return("Success")

def test_2_8(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.shape == (3, 3), "The size of your answer is incorrect. Are you dropping the target column?"
    assert 'target' not in answer.columns, "Make sure you are not selecting the target column as part of your answer."
    return("Success")

def test_2_9(answer):
    assert 'predicted' in answer.columns, "Make sure you are naming the columns with the predictions as 'predicted'."
    assert list(answer['predicted']).count('happy') == 1, "Some predicted values are incorrect. Are you fitting the model and predicting correctly?"
    assert list(answer['predicted']).count('unhappy') == 2, "Some predicted values are incorrect. Are you fitting the model and predicting correctly?" 
    return("Success")

def test_3_1(answer):
    assert sha1(str(answer).encode('utf8')).hexdigest() == "381d0617aa4a47b800653fabafdd9d0d3f5ad2ca", "Your answer is incorrect. Are you looking at the output correctly?"
    return("Success")

def test_3_2(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    assert answer.shape == (8, 10), "The dimensions of your answer is incorrect. Are you using the 'summary' function?"
    assert sorted(list(answer.columns))[0:4] == ['attack', 'defense', 'generation', 'hp'], "Your answer contains the incorrect columns. Are you using the 'summary' function?"
    assert [round(x,2) for x in sorted(list(answer['hp']))] == [1.0, 25.53, 50.0, 65.0, 69.26, 80.0, 255.0, 800.0], "Your answer contains the incorrect values. Are you using the 'summary' function?"
    return("Success")

def test_3_3(answer1,answer2,answer3):
    assert isinstance(answer2, altair.vegalite.v4.api.VConcatChart), "the panel object should be instance of `altair` contatinated chart"
    str_fun = inspect.getsource(answer1) 
    assert "return histogram" in str_fun, "Are you returing the correct variable?"
    assert sha1(str(len(answer3)).encode('utf8')).hexdigest() == "902ba3cda1883801594b6e1b452790cc53948fda", "Your answer is incorrect. Incorrect number of features."
    assert str(set([type(value) for value in answer3.values()])) == "{<class 'altair.vegalite.v4.api.Chart'>}", "Each figure in the dictionary needs to be of type 'altair.vegalite.v4.api.Chart'"
    return("Success")

def test_3_5(answer):
    assert sha1((answer.lower() + 'e').encode('utf8')).hexdigest() == "b452d6b23b3c28f85872fffd99bdaf90ce0ad44a", "Your answer is incorrect. Think about why both might be useful."
    return("Success")

def test_3_6(answer):
    assert sha1((answer.lower() + 'w').encode('utf8')).hexdigest() == "a7ce5b0c7e956b8e3c1a5c254c910d57c56e1b57", "Your answer is incorrect. Consider how the models are perfroming already."
    return("Success")

def test_4_1(answer1,answer2):
    assert not answer1 is None, "The 'X' variable does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "The 'y' variable does not exist. Have you passed in the correct variable?"
    assert answer1.shape == (800, 8), "The size of 'X' is incorrect. Are you dropping the target column?"
    assert answer2.shape == (800,), "The size of 'y' is incorrect. Are you only selecting the target column?"
    assert 'legendary' not in answer1.columns, "Make sure you are not selecting the target column as part of 'X'"
    return("Success")

def test_4_2(answer):
    assert len(answer) == 15, "Your solution has the incorrect length. Are you iterating over 15 values?"
    res = [round(x,2) for x in sorted(answer)]
    assert sha1(str(res).encode('utf8')).hexdigest() == "1dac59d4acb863476eb8b00aa38d4117957ea44a", "Your answer contains incorrect values. Are you fitting the models properly?"
    return("Success")

def test_4_3(answer):
    assert answer.shape == (15, 2), "Your dataframe dimensions are incorrect. Are you creating the dataframe properly?"
    assert sorted(list(answer.columns)) == ['accuracy', 'max_depth'], "Your dataframe contains incorrect columns. Make sure you are naming the columns as instructed."
    return("Success")

def test_4_4(answer):
    assert not answer is None, "Your answer does not exist. Have you passed in the correct variable?"
    try:
        answer.mark.type == "line", "Your plot is not a bar plot. Make sure you are using the 'mark_bar()' function"
    except AttributeError as error:
        assert answer.mark == "line", "Your plot is not a bar plot. Make sure you are using the 'mark_bar()' function"
    assert 'max_depth' in str(answer.encoding.x.shorthand) or 'max_depth' in str(answer.encoding.x.field), "Make sure you are plotting the `max_depth` variable on the x-axis."
    assert 'accuracy' in str(answer.encoding.y.shorthand) or 'accuracy' in str(answer.encoding.y.field), "Make sure you are plotting the `accuracy` variable. on the y-axix"
    assert not answer.title == 'Undefined', "Make sure you are providing a title for the plot."
    return "Success"
    return("Success")
    return("Success")

def test_5_1(answer1,answer2):
    assert not answer1 is None, "The 'X' variable does not exist. Have you passed in the correct variable?"
    assert not answer2 is None, "The 'y' variable does not exist. Have you passed in the correct variable?"
    assert answer1.shape == (414, 5), "The size of 'X' is incorrect. Are you dropping the target column?"
    assert answer2.shape == (414,), "The size of 'y' is incorrect. Are you only selecting the target column?"
    assert 'price' not in answer1.columns, "Make sure you are not selecting the target column as part of 'X'"
    return("Success")

def test_5_2(answer):
    assert sha1(str(round(answer,2)).encode('utf8')).hexdigest() == "715cf4cc66c307fa0a5e37e6f02176e1e11b1554", "Your answer is incorrect. Are you setting up, predicting and scoring the model properly?"
    return("Success")

    

    

    

    

