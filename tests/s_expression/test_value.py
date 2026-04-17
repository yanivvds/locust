import pandas as pd
import pytest

import config
from s_expression.parser import parse, eval
from utils.answer_comparator import assert_frame_equal_unordered

SEXP = """
    (VALUE 85302NED
        (MSR (D004645))
        (DIM Perioden (2021JJ00 2022JJ00))
        (DIM BestemmingEnSeizoen (T001047))
        (DIM Vakantiekenmerken (T001460))
        (DIM Marges (MW00000))
    )
"""

ANSWER_INDEX = [('x 1 000', 'Totaal vakanties')]
ANSWER_COLS = [('Totaal vakanties', 'Waarde', 'Totaal vakanties', '2021'),
               ('Totaal vakanties', 'Waarde', 'Totaal vakanties', '2022')]
ANSWER_DATA = [[31685.0, 35933.0]]
ANSWER_DF = pd.DataFrame(
    data=ANSWER_DATA,
    index=pd.MultiIndex.from_tuples(ANSWER_INDEX, names=['Unit', 'Measure']),
    columns=pd.MultiIndex.from_tuples(ANSWER_COLS, names=['Bestemming en seizoen', 'Marges', 'Vakantiekenmerken', 'Perioden']),
)

MAPPER_RESULT = {'Vakantiekenmerken': 'Vakantiekenmerken', 'T001460': 'Totaal vakanties', 'T001047': 'Totaal vakanties', 'Perioden': 'Perioden', 'Marges': 'Marges', 'MW00000': 'Waarde', 'MOG0095': 'Ondergrens 95%-interval', 'MBG0095': 'Bovengrens 95%-interval', 'L999996': 'Vakantiebestemming: buitenland', 'L008691': 'Vakantiebestemming: Nederland', 'BestemmingEnSeizoen': 'Bestemming en seizoen', 'A048563': 'Soort vakantie: ontspanning', 'A047372': 'Soort vakantie: eigen recreatieve accom.', 'A047152': 'Vervoerwijze: overige ', 'A047151': 'Vervoerwijze: vliegtuig', 'A047150': 'Vervoerwijze: auto', 'A046406': 'Organisatie: eigen recreatieve accom.', 'A046404': 'Organisatie: op de bonnefooi', 'A046403': 'Organisatie: alleen vervoer', 'A046402': 'Organisatie: alleen accommodatie', 'A046401': 'Organisatie: samengestelde reis', 'A046400': 'Organisatie: pakketreis', 'A046399': 'Organisatie: accommodatie en vervoer', 'A046397': 'Vakantieduur: 16 of meer dagen', 'A046396': 'Vakantieduur: 9 tot 16 dagen', 'A046395': 'Vakantieduur: 5 tot 9 dagen', 'A046394': 'Vakantieduur: 5 of meer dagen (lang)', 'A046393': 'Vakantieduur: 2 tot 5 dagen (kort)', 'A046391': 'Soort vakantie: overig, onbekend', 'A046390': 'Soort vakantie: familie, vrienden, e.d.', 'A046389': 'Soort vakantie: rondreis', 'A046388': 'Soort vakantie: recreatie', 'A046387': 'Soort vakantie: cultuur', 'A046386': 'Soort vakantie: natuur', 'A046385': 'Soort vakantie: steden', 'A046384': 'Soort vakantie: strand', 'A046378': 'Soort vakantie: actief', 'A046372': 'Reisgezelschap: vrienden, bekenden, e.a.', 'A046371': 'Reisgezelschap: familie, anderen', 'A046370': 'Reisgezelschap: volwassene en kind(eren)', 'A046369': 'Reisgezelschap: paar zonder kinderen', 'A046368': 'Reisgezelschap: alleenreizende', 'A046366': 'Accommodatie: overig, onbekend', 'A046363': 'Accommodatie: particulier (alle)', 'A046362': 'Accommodatie: kampeerverblijf', 'A046361': 'Accommodatie: appartement', 'A046360': 'Accommodatie: vakantiehuis, stacaravan', 'A046359': 'Accommodatie: hotel, pension, B&B', 'A045879': 'Winterseizoen', 'A045878': 'Zomerseizoen', 'A018981': 'Vervoerwijze: trein', '2023JJ00': '2023', '2022JJ00': '2022', '2021JJ00': '2021', 'M006089_2': 'Totaal vakantie-uitgaven', 'M006089_1': 'Vakantie-uitgaven', 'M005005_3': 'Vakanties', 'M005005_2': 'Percentage Nederlanders', 'M005005_1': 'Totaal Nederlanders', 'M004995': 'Gemiddeld per Nederlander', 'M004993': 'Gemiddeld per deelnemer', 'M004957': 'Gemiddeld per Nederlander', 'M004417': 'Gemiddeld per deelnemer', 'M003924': 'Gemiddeld per deelnemer', 'M003533_2': 'Totaal vakantiedagen', 'M003533_1': 'Vakantiedagen', 'M002336': 'Gemiddeld per vakantie', 'M002334': 'Gemiddeld per Nederlander', 'M001963': 'Gemiddeld per persoon per vakantiedag', 'M001957': 'Gemiddeld per vakantie', 'M001860': 'Gemiddeld per Nederlander', 'M001844': 'Gemiddeld per deelnemer', 'M001339': 'Gemiddeld per persoon per vakantie', 'D006211_2': 'Totaal overnachtingen', 'D006211_1': 'Overnachtingen', 'D004645': 'Totaal vakanties', 'D000084': 'Deelname', '85302NED': 'Vakanties van Nederlanders; kerncijfers'}

@pytest.mark.skipif(config.ENV == 'devops', reason="Test for local use only")
def test_value_sexp():
    """For this test an internet connection is required and the OData API must be live"""
    sexp, answer = eval(parse(SEXP))
    assert_frame_equal_unordered(answer, ANSWER_DF)
    assert sexp.mapper == MAPPER_RESULT

def test_value_sexp_offline():
    sexp, answer = eval(parse(SEXP), offline=True)
    assert_frame_equal_unordered(answer, ANSWER_DF)
    assert sexp.mapper == MAPPER_RESULT

def test_value_odata3_sql():
    sexp, answer = eval(parse(SEXP), sql=True)
    assert_frame_equal_unordered(answer, ANSWER_DF)
    assert sexp.mapper == MAPPER_RESULT

def test_value_simplified_sql():
    expected_answer = pd.DataFrame({'Bestemming en seizoen': ['Totaal vakanties'] * 2, 'Perioden': ['2021', '2022'],
                                    'Marges': ['Waarde'] * 2, 'Vakantiekenmerken': ['Totaal vakanties'] * 2,
                                    'Totaal vakanties': [31685.0, 35933.0]})

    # Check for identical tables with different ordering
    expected_answer_changed_order = pd.DataFrame({'Bestemming en seizoen': ['Totaal vakanties'] * 2, 'Perioden': ['2022', '2021'],
                                    'Marges': ['Waarde'] * 2, 'Vakantiekenmerken': ['Totaal vakanties'] * 2,
                                    'Totaal vakanties': [35933.0, 31685.0]})

    sexp, answer = eval(parse(SEXP), sql=True, simplified=True)
    assert_frame_equal_unordered(answer, expected_answer)
    assert_frame_equal_unordered(answer, expected_answer_changed_order)
    assert sexp.mapper == MAPPER_RESULT


COND_SEXP = """
    (VALUE 85302NED
        (MSR (D004645 < 35000))
        (DIM Perioden (2021JJ00 2022JJ00 2023JJ00 2024JJ00))
        (DIM BestemmingEnSeizoen (T001047))
        (DIM Vakantiekenmerken (T001460))
        (DIM Marges (MW00000))
    )
"""

COND_ANSWER_INDEX = [('x 1 000', 'Totaal vakanties')]
COND_ANSWER_COLS = [('Totaal vakanties', 'Waarde', 'Totaal vakanties', '2021')]
COND_ANSWER_DATA = [[31685.0]]
COND_ANSWER_DF = pd.DataFrame(
    data=COND_ANSWER_DATA,
    index=pd.MultiIndex.from_tuples(COND_ANSWER_INDEX, names=['Unit', 'Measure']),
    columns=pd.MultiIndex.from_tuples(COND_ANSWER_COLS, names=['Bestemming en seizoen', 'Marges', 'Vakantiekenmerken', 'Perioden']),
)

def test_value_sexp_conditional_offline():
    _, answer = eval(parse(COND_SEXP), sql=True)
    assert_frame_equal_unordered(answer, COND_ANSWER_DF)
