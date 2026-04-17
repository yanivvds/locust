import pytest

import config
from s_expression.parser import parse, eval
from utils.custom_types import UnitCompatibilityError

# == UNIT TESTS FOR JOIN OVER DIFFERENT MEASURES UNITS  ==
# M004367 has unit qudt:KiloGM-PER-KiloM
# M004369 has unit qudt:TONNE
FAIL_ON_UNITS_SEXP = """
    (SUM ()
        (VALUE 84957NED
            (MSR (M004367 M004369))
            (DIM Perioden (2015JJ00 2016JJ00 2017JJ00))
            (DIM Vervoerstromen (T001448))
        )
    )
"""

@pytest.mark.skipif(config.ENV == 'devops', reason="Test for local use only")
def test_unit_compatibility_sexp():
    with pytest.raises(UnitCompatibilityError,
                       match="Trying to aggregate over measures with a different unit"):
        eval(parse(FAIL_ON_UNITS_SEXP))

def test_unit_compatibility_sexp_offline():
    with pytest.raises(UnitCompatibilityError,
                       match="Trying to aggregate over measures with a different unit"):
        eval(parse(FAIL_ON_UNITS_SEXP), offline=True)

def test_unit_compatibility_odata3_sql():
    with pytest.raises(UnitCompatibilityError,
                       match="Trying to aggregate over measures with a different unit"):
        eval(parse(FAIL_ON_UNITS_SEXP), sql=True)


# == UNIT TESTS FOR AVG OVER DIMENSIONLESS UNITS  ==
FAIL_ON_DIMENSIONLESS_SEXP = """
    (AVG ()
        (VALUE 85302NED
            (MSR (D004645 M005005_2))
            (DIM Perioden (2021JJ00 2022JJ00))
            (DIM BestemmingEnSeizoen (L008691 L999996))
            (DIM Vakantiekenmerken (T001460))
            (DIM Marges (MW00000))
        )
    )
"""

def test_unit_dimensionless_sexp():
    with pytest.raises(UnitCompatibilityError,
                       match="Trying to aggregate over two or more measures with a dimensionless unit"):
        eval(parse(FAIL_ON_DIMENSIONLESS_SEXP))

def test_unit_dimensionless_sexp_offline():
    with pytest.raises(UnitCompatibilityError,
                       match="Trying to aggregate over two or more measures with a dimensionless unit"):
        eval(parse(FAIL_ON_DIMENSIONLESS_SEXP), offline=True)

def test_unit_dimensionless_odata3_sql():
    with pytest.raises(UnitCompatibilityError,
                       match="Trying to aggregate over two or more measures with a dimensionless unit"):
        eval(parse(FAIL_ON_DIMENSIONLESS_SEXP), sql=True)


# == UNIT TESTS FOR SUM OVER DIFFERENT CONVERTION RATIOS  ==
FAIL_ON_CONVERSION_SEXP = """
    (SUM ()
        (VALUE 85302NED
            (MSR (D004645 D006211_2))
            (DIM Perioden (2021JJ00 2022JJ00))
            (DIM BestemmingEnSeizoen (L008691 L999996))
            (DIM Vakantiekenmerken (T001460))
            (DIM Marges (MW00000))
        )
    )
"""

def test_conversion_compatibility_odata3_sql():
    with pytest.raises(UnitCompatibilityError,
                       match="Trying to aggregate over multiple measures with different conversion multipliers"):
        eval(parse(FAIL_ON_CONVERSION_SEXP), sql=True)
