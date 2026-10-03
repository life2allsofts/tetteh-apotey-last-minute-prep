# Data Pipeline

This directory holds every dataset that the Last-Minute-Prep pipeline consumes or produces.

## Folder Layout

| Folder | Visibility | Purpose |
|--------|-----------|---------|
| `raw/` | 🔒 **private** (gitignored) | Original course notes: `.docx` and `.pdf` files, organised by topic |
| `interim/` | 🔒 private | Machine-extracted questions with no labels (`questions_raw.csv`) |
| `processed/` | Mixed | `questions.csv` (private, hand-labelled) + `questions_sample.csv` (public, synthetic) |

## The `raw/` Folder

Contains the original refresher-course material. **Never commit these files** — they may be copyrighted by the course provider and can contain personal annotations.

Recognised topic subfolders (used as topic hints by `src/extract_questions.py`):

- `LAND LAW/`
- `Land Dispute/`
- `TECHNICAL GUIDELINES/`
- `LS-COURSE-NOTES/`
- `Ethics/`
- `ORAL INTERVIEW/`
- `PHOTOGRAMMETRY/`
- `INSTRUMENTATION...` (any file starting with `INSTRUMENTATION`)
- Loose files at the top of `raw/` are also picked up.

## The `processed/questions.csv` File

**This is the source of truth.** One row per question.

| Column | Type | Description |
|--------|------|-------------|
| `question_text` | string | The full question |
| `topic` | categorical | One of: `Land Law`, `Survey Act`, `Regulations (LI 1444)`, `Technical Guidelines`, `Land Disputes`, `Ethics`, `Instrumentation`, `Photogrammetry`, `Other` |
| `difficulty_label` | categorical | One of: `Easy`, `Medium`, `Hard` |
| `source_file` | string | Relative path of the origin file (set automatically) |
| `notes` | string | Optional free text |

### How to Add More Questions

1. Open `processed/questions.csv` in Excel / Numbers / Google Sheets.
2. Add a new row. Fill in `question_text`, `topic`, and `difficulty_label`.
3. Save as CSV.
4. Re-run `bash scripts/run_training.sh`.

No code changes required.

## Labelling Guidance

- **Easy** — recall a definition or a single rule.
- **Medium** — apply a rule to a scenario.
- **Hard** — cross-reference two statutes, analyse a case, or reason about a dispute.