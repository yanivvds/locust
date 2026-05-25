# sidequests/

Exploratory retriever variants that are **not** part of the main thesis pipeline.

Code in this folder exists to test side hypotheses (e.g. "does CRUSH4SQL-style query hallucination help LOCuST retrieval?") without polluting the core retriever set in `models/retrievers/`. Treat anything here as throwaway unless explicitly promoted out of this folder.

Conventions:
- Outputs (caches, result JSONs) go to `evaluation/results/sidequests/`, not the main results folder.
- Each retriever should document its experimental scope in its module docstring.
- Nothing in `sidequests/` should be imported by code outside `sidequests/`.
