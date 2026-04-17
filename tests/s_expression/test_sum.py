import pandas as pd
import pytest

import config
from s_expression.parser import parse, eval
from utils.answer_comparator import assert_frame_equal_unordered

# == UNIT TESTS FOR JOIN OVER MEASURES ==
SUM_ON_MSR_SEXP = """
    (SUM ()
        (VALUE 80781ned
            (MSR (A028997_1 A028994_1))
            (DIM Perioden (2020JJ00 2021JJ00 2022JJ00))
            (DIM RegioS (NL01))
        )
    )
"""

SUM_ON_MSR_ANSWER_INDEX_SEXP = ['Biggen\nLeghennen']
SUM_ON_MSR_ANSWER_INDEX_SQL = [('Biggen;Leghennen', '')]
SUM_ON_MSR_ANSWER_COLS = [('Nederland', '2020'), ('Nederland','2021'), ('Nederland','2022')]
SUM_ON_MSR_ANSWER_DATA = [[48579668., 48329610., 47359417.]]
SUM_ON_MSR_ANSWER_DF_SEXP = pd.DataFrame(
    data=SUM_ON_MSR_ANSWER_DATA,
    index=pd.Index(SUM_ON_MSR_ANSWER_INDEX_SEXP),
    columns=pd.MultiIndex.from_tuples(SUM_ON_MSR_ANSWER_COLS, names=['Regio', 'Perioden']),
)
SUM_ON_MSR_ANSWER_DF_SQL = pd.DataFrame(
    data=SUM_ON_MSR_ANSWER_DATA,
    index=pd.Index(SUM_ON_MSR_ANSWER_INDEX_SQL),
    columns=pd.MultiIndex.from_tuples(SUM_ON_MSR_ANSWER_COLS, names=['Regio', 'Perioden']),
)
SUM_ON_MSR_ANSWER_DF_SQL.index.set_names('Measure', level=0, inplace=True)
SUM_ON_MSR_ANSWER_DF_SQL.index.set_names('Unit', level=1, inplace=True)

@pytest.mark.skipif(config.ENV == 'devops', reason="Test for local use only")
def test_sum_on_msr_sexp():
    """For this test an internet connection is required and the OData API must be live"""
    _, answer = eval(parse(SUM_ON_MSR_SEXP))
    assert_frame_equal_unordered(answer, SUM_ON_MSR_ANSWER_DF_SEXP)

def test_sum_on_msr_sexp_offline():
    _, answer = eval(parse(SUM_ON_MSR_SEXP), offline=True)
    assert_frame_equal_unordered(answer, SUM_ON_MSR_ANSWER_DF_SEXP)

def test_sum_on_msr_odata3_sql():
    _, answer = eval(parse(SUM_ON_MSR_SEXP), sql=True)
    assert_frame_equal_unordered(answer, SUM_ON_MSR_ANSWER_DF_SQL)

def test_sum_on_msr_simplified_sql():
    expected_answer = pd.DataFrame({'Perioden': ['2020', '2021', '2022'],
                                    "Regio's": ['Nederland'] * 3,
                                    'SUM': [48579668., 48329610., 47359417.]}, index=range(3))
    _, answer = eval(parse(SUM_ON_MSR_SEXP), sql=True, simplified=True)
    assert_frame_equal_unordered(answer, expected_answer)


# == UNIT TESTS FOR SUM OVER DIMENSION GROUP ==
SUM_ON_DIM_SEXP = """
    (SUM
        (Perioden)
        (VALUE 85302NED
            (MSR (D004645 M005005_1))
            (DIM Perioden (2021JJ00 2022JJ00))
            (DIM BestemmingEnSeizoen (L008691 L999996))
            (DIM Vakantiekenmerken (T001460))
            (DIM Marges (MW00000))
        )
    )
"""

SUM_ON_DIM_ANSWER_INDEX = [('Totaal Nederlanders', 'x 1 000'), ('Totaal vakanties', 'x 1 000')]
SUM_ON_DIM_ANSWER_COLS = [('Vakantiebestemming: Nederland', 'Waarde', 'Totaal vakanties'),
                          ('Vakantiebestemming: buitenland', 'Waarde', 'Totaal vakanties')]
SUM_ON_DIM_ANSWER_DATA = [[16749.0, 15670.0],
                          [37938.0, 29681.0]]
SUM_ON_DIM_ANSWER_DF = pd.DataFrame(
    data=SUM_ON_DIM_ANSWER_DATA,
    index=pd.Index(SUM_ON_DIM_ANSWER_INDEX).set_names(['Measure', 'Unit']),
    columns=SUM_ON_DIM_ANSWER_COLS,
)

@pytest.mark.skipif(config.ENV == 'devops', reason="Test for local use only")
def test_sum_on_dim_sexp():
    """For this test an internet connection is required and the OData API must be live"""
    sum_cols = [('Vakantiebestemming: Nederland', 'Waarde', 'Totaal vakanties', 'SUM[\'Perioden\']'),
                ('Vakantiebestemming: buitenland', 'Waarde', 'Totaal vakanties', 'SUM[\'Perioden\']')]
    df = SUM_ON_DIM_ANSWER_DF.copy()
    df.columns = sum_cols
    _, answer = eval(parse(SUM_ON_DIM_SEXP))
    assert_frame_equal_unordered(answer, df)

def test_sum_on_dim_sexp_offline():
    sum_cols = [('Vakantiebestemming: Nederland', 'Waarde', 'Totaal vakanties', 'SUM[\'Perioden\']'),
                ('Vakantiebestemming: buitenland', 'Waarde', 'Totaal vakanties', 'SUM[\'Perioden\']')]
    df = SUM_ON_DIM_ANSWER_DF.copy()
    df.columns = sum_cols
    _, answer = eval(parse(SUM_ON_DIM_SEXP), offline=True)
    assert_frame_equal_unordered(answer, df)

def test_sum_on_dim_odata3_sql():
    _, answer = eval(parse(SUM_ON_DIM_SEXP), sql=True)
    assert_frame_equal_unordered(answer, SUM_ON_DIM_ANSWER_DF)

def test_sum_on_dim_simplified_sql():
    expected_answer = pd.DataFrame({'Perioden': ['2021', '2021', '2022', '2022'],
                                    'Bestemming en seizoen': ['Vakantiebestemming: Nederland', 'Vakantiebestemming: buitenland'] * 2,
                                    'Marges': ['Waarde'] * 4,
                                    'Vakantiekenmerken': ['Totaal vakanties'] * 4,
                                    'SUM': [30397., 16743., 24290., 28608.]}, index=range(4))
    _, answer = eval(parse(SUM_ON_DIM_SEXP), sql=True, simplified=True)
    assert_frame_equal_unordered(answer, expected_answer)


# == UNIT TESTS FOR SUM OVER COMPLEX INNER EXPRESSION ==
SUM_ON_JOIN_SEXP = """
    (SUM ()
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
    )
"""

SUM_ON_JOIN_ANSWER_INDEX = [('Ladingtonkilometer', 'mln tonkm')]
SUM_ON_JOIN_ANSWER_COLS = ['SUM']
SUM_ON_JOIN_ANSWER_DATA = [[57987.0]]
SUM_ON_JOIN_ANSWER_DF = pd.DataFrame(
    data=SUM_ON_JOIN_ANSWER_DATA,
    index=pd.Index(SUM_ON_JOIN_ANSWER_INDEX).set_names(['Measure', 'Unit']),
    columns=pd.Index(SUM_ON_JOIN_ANSWER_COLS),
)

@pytest.mark.skipif(config.ENV == 'devops', reason="Test for local use only")
def test_sum_on_join_sexp():
    """For this test an internet connection is required and the OData API must be live"""
    _, answer = eval(parse(SUM_ON_JOIN_SEXP))
    df = SUM_ON_JOIN_ANSWER_DF.copy()
    df.index = pd.Index(["('Ladingtonkilometer', 'mln tonkm')"], name='Measure')
    df.columns = pd.Index([('Ladingtonkilometer', 'mln tonkm')])
    assert_frame_equal_unordered(answer, df)

def test_sum_on_join_sexp_offline():
    _, answer = eval(parse(SUM_ON_JOIN_SEXP), offline=True)
    df = SUM_ON_JOIN_ANSWER_DF.copy()
    df.index = pd.Index(["('Ladingtonkilometer', 'mln tonkm')"], name='Measure')
    df.columns = pd.Index([('Ladingtonkilometer', 'mln tonkm')])
    assert_frame_equal_unordered(answer, df)

def test_sum_on_join_odata3_sql():
    _, answer = eval(parse(SUM_ON_JOIN_SEXP), sql=True)
    assert_frame_equal_unordered(answer, SUM_ON_JOIN_ANSWER_DF)
