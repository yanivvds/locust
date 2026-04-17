import pandas as pd
import pytest

import config
from s_expression.parser import parse, eval
from utils.answer_comparator import assert_frame_equal_unordered

AVG_SEXP = """
    (AVG
        (Perioden)
        (VALUE 85302NED
            (MSR (D004645))
            (DIM Perioden (2021JJ00 2022JJ00))
            (DIM BestemmingEnSeizoen (T001047))
            (DIM Vakantiekenmerken (T001460))
            (DIM Marges (MW00000))
        )
    )
"""

AVG_ANSWER_DATA = [[33809.0]]
AVG_ANSWER_INDEX_SEXP = [('x 1 000', 'Totaal vakanties')]
AVG_ANSWER_COLS_SEXP = [('Totaal vakanties', 'Waarde', 'Totaal vakanties', "AVG['Perioden']")]
AVG_ANSWER_DF_SEXP = pd.DataFrame(
    data=AVG_ANSWER_DATA,
    index=pd.MultiIndex.from_tuples(AVG_ANSWER_INDEX_SEXP, names=['Unit', 'Measure']),
    columns=pd.MultiIndex.from_tuples(AVG_ANSWER_COLS_SEXP, names=['Bestemming en seizoen', 'Marges', 'Vakantiekenmerken', 'Perioden']),
)

AVG_ANSWER_INDEX_SQL = [('Totaal vakanties', 'x 1 000')]
AVG_ANSWER_COLS_SQL = [('Totaal vakanties', 'Waarde', 'Totaal vakanties')]
AVG_ANSWER_DF_SQL = pd.DataFrame(
    data=AVG_ANSWER_DATA,
    index=pd.MultiIndex.from_tuples(AVG_ANSWER_INDEX_SQL, names=['Measure', 'Unit']),
    columns=pd.MultiIndex.from_tuples(AVG_ANSWER_COLS_SQL, names=['Bestemming en seizoen', 'Marges', 'Vakantiekenmerken']),
)

@pytest.mark.skipif(config.ENV == 'devops', reason="Test for local use only")
def test_avg_sexp():
    """For this test an internet connection is required and the OData4 API must be live"""
    _, answer = eval(parse(AVG_SEXP))
    assert_frame_equal_unordered(answer, AVG_ANSWER_DF_SEXP)

def test_avg_sexp_offline():
    _, answer = eval(parse(AVG_SEXP), offline=True)
    assert_frame_equal_unordered(answer, AVG_ANSWER_DF_SEXP)

def test_avg_odata3_sql():
    _, answer = eval(parse(AVG_SEXP), sql=True, odata4=False)
    assert_frame_equal_unordered(answer, AVG_ANSWER_DF_SQL)

def test_avg_simplify_sql():
    expected_answer = pd.DataFrame({'AVG': [31685., 35933.],
                                    'Bestemming en seizoen': ['Totaal vakanties'] * 2,
                                    'Marges': ['Waarde'] * 2, 'Perioden': ['2021', '2022'],
                                    'Vakantiekenmerken': ['Totaal vakanties'] * 2}, index=range(2))
    _, answer = eval(parse(AVG_SEXP), sql=True, simplified=True)
    assert_frame_equal_unordered(answer, expected_answer)
