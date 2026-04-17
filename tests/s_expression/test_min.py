import pandas as pd
import pytest

import config
from s_expression.parser import parse, eval
from utils.answer_comparator import assert_frame_equal_unordered

MIN_ON_DIM_SEXP = """
    (MIN
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

# S-exp returns (argmin, min)
MIN_ON_MSR_ANSWER_DATA_SEXP = [[('2021', 31685.0)]]
MIN_ON_DIM_ANSWER_INDEX_SEXP = [('x 1 000', 'Totaal vakanties')]
MIN_ON_DIM_ANSWER_COLS_SEXP = [('Totaal vakanties', 'Waarde', 'Totaal vakanties', 'MIN[\'Perioden\']')]
MIN_ON_DIM_ANSWER_DF_SEXP = pd.DataFrame(
    data=MIN_ON_MSR_ANSWER_DATA_SEXP,
    index=pd.MultiIndex.from_tuples(MIN_ON_DIM_ANSWER_INDEX_SEXP, names=['Unit', 'Measure']),
    columns=pd.Index(MIN_ON_DIM_ANSWER_COLS_SEXP),
)

@pytest.mark.skipif(config.ENV == 'devops', reason="Test for local use only")
def test_min_on_dim_sexp():
    """For this test an internet connection is required and the OData API must be live"""
    _, answer = eval(parse(MIN_ON_DIM_SEXP))
    assert_frame_equal_unordered(answer, MIN_ON_DIM_ANSWER_DF_SEXP)

def test_min_on_dim_sexp_offline():
    _, answer = eval(parse(MIN_ON_DIM_SEXP), offline=True)
    assert_frame_equal_unordered(answer, MIN_ON_DIM_ANSWER_DF_SEXP)

# SQL returns just min
MIN_ON_DIM_ANSWER_DATA_SQL = [[31685.0]]
MIN_ON_DIM_ANSWER_INDEX_SQL = [('Totaal vakanties', 'x 1 000')]
MIN_ON_DIM_ANSWER_COLS_SQL = [('Totaal vakanties', 'Waarde', 'Totaal vakanties')]
MIN_ON_DIM_ANSWER_DF_SQL = pd.DataFrame(
    data=MIN_ON_DIM_ANSWER_DATA_SQL,
    index=pd.MultiIndex.from_tuples(MIN_ON_DIM_ANSWER_INDEX_SQL, names=['Measure', 'Unit']),
    columns=pd.MultiIndex.from_tuples(MIN_ON_DIM_ANSWER_COLS_SQL, names=['Bestemming en seizoen', 'Marges', 'Vakantiekenmerken']),
)

def test_min_on_dim_odata3_sql():
    _, answer = eval(parse(MIN_ON_DIM_SEXP), sql=True)
    assert_frame_equal_unordered(answer, MIN_ON_DIM_ANSWER_DF_SQL)

def test_min_on_dim_simplified_sql():
    # Simplified SQL returns (argmin, min)
    expected_answer = pd.DataFrame({'Vakantiekenmerken': 'Totaal vakanties', 'Marges': 'Waarde',
                                    'Bestemming en seizoen': 'Totaal vakanties', 'MIN[Perioden]': '2021',
                                    'Totaal vakanties': 31685.0}, index=range(1))
    _, answer = eval(parse(MIN_ON_DIM_SEXP), sql=True, simplified=True)
    assert_frame_equal_unordered(answer, expected_answer)


MIN_ON_MSR_SEXP = """
    (MIN
        ()
        (VALUE 85302NED
            (MSR (D004645 D006211_2))
            (DIM Perioden (2021JJ00 2022JJ00))
            (DIM BestemmingEnSeizoen (T001047))
            (DIM Vakantiekenmerken (T001460))
            (DIM Marges (MW00000))
        )
    )
"""

# S-exp returns (argmin, min)
MIN_ON_MSR_ANSWER_DATA_SEXP = [[('Totaal vakanties', 'aantal', 31685000.0),
                                ('Totaal vakanties', 'aantal', 35933000.0)]]
MIN_ON_MSR_ANSWER_INDEX_SEXP = ['Totaal overnachtingen\nTotaal vakanties']
MIN_ON_MSR_ANSWER_COLS_SEXP = [('Totaal vakanties', 'Waarde', 'Totaal vakanties', '2021'),
                               ('Totaal vakanties', 'Waarde', 'Totaal vakanties', '2022')]
MIN_ON_MSR_ANSWER_DF_SEXP = pd.DataFrame(
    data=MIN_ON_MSR_ANSWER_DATA_SEXP,
    index=pd.Index(MIN_ON_MSR_ANSWER_INDEX_SEXP),
    columns=pd.Index(MIN_ON_MSR_ANSWER_COLS_SEXP),
)

@pytest.mark.skipif(config.ENV == 'devops', reason="Test for local use only")
def test_min_on_msr_sexp():
    """For this test an internet connection is required and the OData API must be live"""
    _, answer = eval(parse(MIN_ON_MSR_SEXP))
    assert_frame_equal_unordered(answer, MIN_ON_MSR_ANSWER_DF_SEXP)

def test_min_on_msr_sexp_offline():
    _, answer = eval(parse(MIN_ON_MSR_SEXP), offline=True)
    assert_frame_equal_unordered(answer, MIN_ON_MSR_ANSWER_DF_SEXP)

# SQL returns just min
MIN_ON_MSR_SQL = """
    (MIN
        ()
        (VALUE 85302NED
            (MSR (M001339 M001860))
            (DIM Perioden (2021JJ00 2022JJ00))
            (DIM BestemmingEnSeizoen (T001047))
            (DIM Vakantiekenmerken (T001460))
            (DIM Marges (MW00000))
        )
    )
"""
MIN_ON_MSR_ANSWER_DATA_SQL = [[446., 571.]]
MIN_ON_MSR_ANSWER_INDEX_SQL = [('Gemiddeld per persoon per vakantie;Gemiddeld per Nederlander', '')]
MIN_ON_MSR_ANSWER_COLS_SQL = [('2021', 'Totaal vakanties', 'Waarde', 'Totaal vakanties'),
                              ('2022', 'Totaal vakanties', 'Waarde', 'Totaal vakanties')]
MIN_ON_MSR_ANSWER_DF_SQL = pd.DataFrame(
    data=MIN_ON_MSR_ANSWER_DATA_SQL,
    index=pd.MultiIndex.from_tuples(MIN_ON_MSR_ANSWER_INDEX_SQL, names=['Measure', 'Unit']),
    columns=pd.MultiIndex.from_tuples(MIN_ON_MSR_ANSWER_COLS_SQL, names=['Perioden', 'Bestemming en seizoen', 'Marges', 'Vakantiekenmerken']),
)

def test_min_on_msr_odata3_sql():
    _, answer = eval(parse(MIN_ON_MSR_SQL), sql=True)
    assert_frame_equal_unordered(answer, MIN_ON_MSR_ANSWER_DF_SQL)

def test_min_on_msr_simplified_sql():
    # Simplified SQL returns (argmin, min)
    expected_answer = pd.DataFrame({'Vakantiekenmerken': 'Totaal vakanties', 'Marges': 'Waarde',
                                    'Bestemming en seizoen': 'Totaal vakanties', 'MIN[Value]': 446.,
                                    'Measure': 'Gemiddeld per persoon per vakantie', 'Perioden': '2021'}, index=range(1))
    _, answer = eval(parse(MIN_ON_MSR_SQL), sql=True, simplified=True)
    assert_frame_equal_unordered(answer, expected_answer)
