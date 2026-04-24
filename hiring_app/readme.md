# HireIQ — AI Candidate Matching

## How to Run

**Requirements:** Python 3.11+

1. Install dependencies:
   ```
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. Start the server:
   ```
   cd hiring_app
   py -3.11 app.py
   ```

3. Open your browser at `http://localhost:5000`

## API

**POST** `/recommend`

```json
{
  "job_description": "Looking for a Python developer with ML experience...",
  "top_n": 5
}
```

Returns ranked candidates with `matched_skills`, `missing_skills`, `explanation`, and `total_score`.

## Running Tests

```
cd hiring_app
pytest test_pipeline.py -v
```
