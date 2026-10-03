# Trained Models

Two serialised TabPFN classifiers live here after training:

| File | Predicts |
|------|----------|
| `topic_model.joblib` | The `topic` label of a question |
| `difficulty_model.joblib` | The `difficulty_label` of a question |

## Regenerating

```bash
bash scripts/run_training.sh

When to Retrain
You added or edited rows in data/processed/questions.csv.

You changed the feature set in src/feature_engineering.py.

You bumped the TabPFN version in requirements.txt.

Versioning
.joblib files are gitignored — they are regenerable artefacts, not source. Keep a local backup if you care about reproducing a specific result.


---

## 🛠️ Step 3 — Source Code (`src/`)

### `src/__init__.py`

```python
"""Last-Minute-Prep: offline-first TabPFN study companion."""

__version__ = "0.1.0"