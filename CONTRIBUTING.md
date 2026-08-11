# Contributing

Contributions that improve reproducibility, evaluation, documentation, and modern Python compatibility are welcome.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Pull requests

- Keep legacy Python 2 files available for historical reference unless a migration includes a documented replacement.
- Add tests for command-line or model-format changes.
- Do not commit locally trained models, virtual environments, credentials, or private datasets.
- Report evaluation results with the random seed, split strategy, data version, and metric definition.
- Document any new data source, its license, and its privacy constraints.

## Model changes

A higher score on the bundled data is not sufficient on its own. Explain possible leakage, class balance, domain assumptions, and how the model behaves on unseen data.
