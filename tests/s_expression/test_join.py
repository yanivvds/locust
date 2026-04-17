import pandas as pd
import pytest

import config
from s_expression.parser import parse, eval
from utils.answer_comparator import assert_frame_equal_unordered

# == UNIT TESTS FOR JOIN ON DIMENSION GROUP ==
# Let's ignore that joining these two tables doesn't make any sense
JOIN_ON_DIM_SEXP = """
    (JOIN (Perioden)
        (VALUE 84957NED
            (MSR (M004367))
            (DIM Perioden (2021JJ00 2022JJ00))
            (DIM Vervoerstromen (A045747))
        )
        (VALUE 85302NED
            (MSR (D006211_2))
            (DIM Perioden (2021JJ00 2022JJ00))
            (DIM BestemmingEnSeizoen (T001047))
            (DIM Vakantiekenmerken (T001460))
            (DIM Marges (MW00000))
        )
    )
"""

JOIN_ON_DIM_ANSWER_DATA = [[11003., 205.2], [9659., 265.4]]
JOIN_ON_DIM_ANSWER_INDEX_SEXP = ['2021', '2022']
JOIN_ON_DIM_ANSWER_COLS_SEXP = [('Ladingtonkilometer', 'Binnenlands', 'mln tonkm'), ('Totaal overnachtingen', 'Totaal vakanties', 'Waarde')]
JOIN_ON_DIM_ANSWER_DF_SEXP = pd.DataFrame(
    data=JOIN_ON_DIM_ANSWER_DATA,
    index=pd.Index(JOIN_ON_DIM_ANSWER_INDEX_SEXP),
    columns=pd.MultiIndex.from_tuples(JOIN_ON_DIM_ANSWER_COLS_SEXP),
)

JOIN_ON_DIM_ANSWER_INDEX_SQL = ['2021', '2022']
JOIN_ON_DIM_ANSWER_COLS_SQL = [('Ladingtonkilometer', 'Binnenlands', '', ''), ('Totaal vakanties', 'Totaal overnachtingen', 'Waarde', 'Totaal vakanties')]
JOIN_ON_DIM_ANSWER_DF_SQL = pd.DataFrame(
    data=JOIN_ON_DIM_ANSWER_DATA,
    index=pd.Index(JOIN_ON_DIM_ANSWER_INDEX_SQL),
    columns=pd.MultiIndex.from_tuples(JOIN_ON_DIM_ANSWER_COLS_SQL),
)

@pytest.mark.skipif(config.ENV == 'devops', reason="Test for local use only")
def test_join_on_dim_sexp():
    """For this test an internet connection is required and the OData API must be live"""
    _, answer = eval(parse(JOIN_ON_DIM_SEXP))
    assert_frame_equal_unordered(answer, JOIN_ON_DIM_ANSWER_DF_SEXP)

@pytest.mark.skip(reason="Offline SQL joins are not yet identical to SEXP joins")
def test_join_on_dim_sexp_offline():
    # FIXME: make sure the results of the offline mode is identical
    #  to the online mode and Units are being returned as well
    expected_columns = [('Ladingtonkilometer', 'Binnenlands'), ('Totaal overnachtingen', 'Totaal vakanties')]
    expected_df = pd.DataFrame(
        data=JOIN_ON_DIM_ANSWER_DATA,
        index=pd.Index(JOIN_ON_DIM_ANSWER_INDEX_SEXP),
        columns=pd.MultiIndex.from_tuples(expected_columns, names=['foo', 'bar']),
    )

    _, answer = eval(parse(JOIN_ON_DIM_SEXP), offline=True)
    assert_frame_equal_unordered(answer, expected_df)

def test_join_on_dim_odata3_sql():
    _, answer = eval(parse(JOIN_ON_DIM_SEXP), sql=True)
    assert_frame_equal_unordered(answer, JOIN_ON_DIM_ANSWER_DF_SQL)


# == UNIT TEST FOR JOIN ON MEASURES ==
# No, it doesn't make any sense to JOIN two of the same tables, but it can test the workings perfectly
JOIN_ON_MSR_SEXP = """
    (JOIN (M004367)
        (VALUE 84957NED
            (MSR (M004367))
            (DIM Perioden (2021JJ00 2022JJ00))
            (DIM Vervoerstromen (A045747))
        )
        (VALUE 84957NED
            (MSR (M004367))
            (DIM Perioden (2016JJ00 2017JJ00))
            (DIM Vervoerstromen (A045747))
        )
    )
"""

JOIN_ON_MSR_ANSWER_DATA = [[11003., 9659., 19379., 17946.]]
JOIN_ON_MSR_ANSWER_INDEX_SEXP = [('Ladingtonkilometer', 'mln tonkm')]
JOIN_ON_MSR_ANSWER_COLS_SEXP = [('Binnenlands', '2021'), ('Binnenlands', '2022'), ('Binnenlands', '2016'), ( 'Binnenlands', '2017')]
JOIN_ON_MSR_ANSWER_DF_SEXP = pd.DataFrame(
    data=JOIN_ON_MSR_ANSWER_DATA,
    index=pd.Index(JOIN_ON_MSR_ANSWER_INDEX_SEXP, tupleize_cols=False),
    columns=pd.MultiIndex.from_tuples(JOIN_ON_MSR_ANSWER_COLS_SEXP),
)

JOIN_ON_MSR_ANSWER_DF_SQL = pd.DataFrame(
    data=JOIN_ON_MSR_ANSWER_DATA,
    index=pd.MultiIndex.from_tuples(JOIN_ON_MSR_ANSWER_INDEX_SEXP, names=['Measure', 'Unit']),
    columns=pd.MultiIndex.from_tuples(JOIN_ON_MSR_ANSWER_COLS_SEXP),
)

@pytest.mark.skipif(config.ENV == 'devops', reason="Test for local use only")
def test_join_on_msr_sexp():
    """For this test an internet connection is required and the OData API must be live"""
    _, answer = eval(parse(JOIN_ON_MSR_SEXP))
    assert_frame_equal_unordered(answer, JOIN_ON_MSR_ANSWER_DF_SEXP)

def test_join_on_msr_sexp_offline():
    _, answer = eval(parse(JOIN_ON_MSR_SEXP), offline=True)
    assert_frame_equal_unordered(answer, JOIN_ON_MSR_ANSWER_DF_SEXP)

def test_join_on_msr_odata3_sql():
    _, answer = eval(parse(JOIN_ON_MSR_SEXP), sql=True)
    assert_frame_equal_unordered(answer, JOIN_ON_MSR_ANSWER_DF_SQL)
