import pandas as pd
import pytest

import config
from s_expression.parser import parse, eval
from utils.answer_comparator import assert_frame_equal_unordered

PROP_SEXP = """
    (PROP
        (DIM Perioden (2021JJ00))
        (VALUE 85302NED
            (MSR (D004645))
            (DIM Perioden (2021JJ00 2022JJ00))
            (DIM BestemmingEnSeizoen (T001047))
            (DIM Vakantiekenmerken (T001460))
            (DIM Marges (MW00000))
        )
    )
"""

PROP_ANSWER_DATA = [[67618., 31685., 46.8588]]

PROP_ANSWER_INDEX_SEXP = [('Totaal vakanties', 'Waarde', 'Totaal vakanties', "PROP['Perioden']")]
PROP_ANSWER_COLS_SEXP = [('Totaal vakanties', 'x 1 000'), ('2021', ''), ('%', '')]
PROP_ANSWER_DF_SEXP = pd.DataFrame(
    data=PROP_ANSWER_DATA,
    index=pd.Index(PROP_ANSWER_INDEX_SEXP).to_flat_index(),
    columns=pd.Index(PROP_ANSWER_COLS_SEXP),
)

PROP_ANSWER_INDEX_SQL = [('Totaal vakanties, Waarde, Totaal vakanties, Totaal vakanties', 'x 1 000')]
PROP_ANSWER_COLS_SQL = ["SUM[Perioden]", '2021', '%']
PROP_ANSWER_DF_SQL = pd.DataFrame(
    data=PROP_ANSWER_DATA,
    index=pd.MultiIndex.from_tuples(PROP_ANSWER_INDEX_SQL, names=['Measure', 'Unit']),
    columns=pd.Index(PROP_ANSWER_COLS_SQL)
)

@pytest.mark.skipif(config.ENV == 'devops', reason="Test for local use only")
def test_prop_sexp():
    """For this test an internet connection is required and the OData API must be live"""
    _, answer = eval(parse(PROP_SEXP))
    assert_frame_equal_unordered(answer, PROP_ANSWER_DF_SEXP)


def test_prop_sexp_offline():
    _, answer = eval(parse(PROP_SEXP), offline=True)
    assert_frame_equal_unordered(answer, PROP_ANSWER_DF_SEXP)


def test_prop_odata3_sql():
    _, answer = eval(parse(PROP_SEXP), sql=True)
    assert_frame_equal_unordered(answer, PROP_ANSWER_DF_SQL, rtol=1e-4)
