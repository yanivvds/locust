import os
import sys
from pathlib import Path

from logs import logging

logger = logging.getLogger(__name__)

PATH_ROOT = Path(__file__).parent.__str__()
PATH_DIR_DATA = Path(Path(__file__).parent, 'data').__str__()
PATH_DIR_GRAPHS = Path(Path(__file__).parent, 'odata_graph/graphs').__str__()

ENV = os.getenv('ENV', 'local')
IS_UNIT_TESTING = 'pytest' in sys.modules

# Language in which to run the application ('nl', 'en')
LANGUAGE = os.getenv('LANGUAGE', 'nl')

# MISC.
ODATA4_BASE_URL = 'https://datasets.cbs.nl/odata/v1/CBS'
ODATA3_BASE_URL = 'https://opendata.cbs.nl/ODataApi/OData'
TQDM_BAR_FMT = '{l_bar}{bar:30}{r_bar}{bar:-30b}'

# GRAPH DB
LOCAL_GRAPH = os.getenv('LOCAL_GRAPH', 'True') != 'False'
GRAPH_DB_HOST = os.getenv('GRAPH_DB_HOST', 'http://localhost:7200')
GRAPH_DB_USERNAME = os.getenv('GRAPH_DB_USERNAME', 'local')
GRAPH_DB_PASSWORD = os.getenv('GRAPH_DB_PASSWORD', None)
GRAPH_DB_REPO = os.getenv('GRAPH_DB_REPO', 'cbs-en' if LANGUAGE == 'en' else 'cbs-nl')
# Only relevant when loading in local file
GRAPH_FILE = (os.getenv('GRAPH_FILE', f"{PATH_DIR_GRAPHS}/{GRAPH_DB_REPO}.trig") if not IS_UNIT_TESTING
              else f"{PATH_DIR_GRAPHS}/ut_graph.trig")

# DATABASE
if IS_UNIT_TESTING:
    DB_ODATA3_FILES = f"{PATH_DIR_DATA}/tests/odata3"
elif LANGUAGE == 'nl':
    DB_ODATA3_FILES = f"{PATH_DIR_DATA}/nl/odata3"
    TRAIN_QA_FILE = f"{PATH_DIR_DATA}/qa_pairs/nl/complex_nl_train.jsonl"
    TEST_QA_FILE = f"{PATH_DIR_DATA}/qa_pairs/nl/complex_nl_test.jsonl"
elif LANGUAGE == 'en':
    DB_ODATA3_FILES = f"{PATH_DIR_DATA}/en/odata3"
    TRAIN_QA_FILE = f"{PATH_DIR_DATA}/qa_pairs/en/complex_en_train.jsonl"
    TEST_QA_FILE = f"{PATH_DIR_DATA}/qa_pairs/en/complex_en_test.jsonl"
else:
    raise ValueError(f"Unsupported language '{LANGUAGE}'. Language must be in [nl, en].")

# AZURE
AZURE_ENDPOINT = "<PLACEHOLDER>"
AZURE_API_VERSION = "<PLACEHOLDER>"
AZURE_KEY = "<PLACEHOLDER>"

logger.info("\n=== Running script with the following configuration ===" +
            f"\n\tLANGUAGE: {LANGUAGE}" +
            f"\n\tLOCAL_GRAPH: {LOCAL_GRAPH}" +
            (f"\n\tGRAPH_FILE: {GRAPH_FILE}" if LOCAL_GRAPH else '') +
            f"\n\tGRAPH_DB_HOST: {GRAPH_DB_HOST}" +
            f"\n\tGRAPH_DB_REPO: {GRAPH_DB_REPO}")
