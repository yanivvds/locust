#!/bin/bash
. venv13/bin/activate

# DEFAULT VARIABLES FOR EXPERIMENTS
export PYTHONPATH=.
export LANGUAGE=en
export LOCAL_GRAPH=False
export GRAPH_DB_HOST=https://vocabs.cbs.nl/graphdb
export GRAPH_DB_USERNAME=<username>
export GRAPH_DB_PASSWORD=<passwqord>

nohup python3.13 models/retrievers/colbert/colbert_trainer.py \
  --mode all \
  --negative_n 10 \
  --output_name colbertv2-english-all-nodes \
  --num_rounds 2 \
  > logs/colbert_trainer_english.log 2>&1 &