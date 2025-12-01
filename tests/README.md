# Tests

This repository uses **pytest** for a lightweight verification of the FastAPI service (`/health` and `/optimize`). The tests in `tests/test_api.py` exercise the live optimizer and validate the response schema.

## Running the suite

```bash
pytest
```

> The `/optimize` test invokes the real optimizer, so allow roughly ten seconds for the run to finish.
