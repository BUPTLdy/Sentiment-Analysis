# Security Policy

## Serialized models

Python pickle and joblib files can execute code while loading. Keras HDF5/YAML artifacts may also depend on unsafe or obsolete deserialization behavior.

- Only load model files you created locally or obtained from a trusted source.
- Do not accept arbitrary model uploads in a service that calls this repository's loaders.
- Retrain the modern baseline from the documented dataset when provenance is uncertain.

## Data and model limitations

The bundled review data is historical and may contain biased, offensive, or privacy-sensitive text. Do not use the example model for consequential decisions without a separate data review, representative evaluation, and appropriate human oversight.

Please report vulnerabilities privately to the repository owner before publishing exploit details.
