# Project Team Selection Dataset

This dataset supports an audited agent decision: choose exactly three people for the
project described in `project_requirements.md`.

- `team_selection.db`: ready-to-query SQLite database.
- `team_selection.sql`: reproducible schema and seed data.
- `project_requirements.md`: constraints and preference order.
- `manager_feedback.md`: qualitative evidence.
- `candidate_profiles.md`: résumé-style summaries.

The database contains exactly 50 people plus normalized skills, availability, experience,
and assignments. A human-checkable reference team is Alice Chen, Ben Ortiz, and Cara Singh.
Their cost is USD 17,280 (`24 hours × 4 weeks × USD 180/hour`).

Rebuild from the repository root:

```powershell
.\.venv\Scripts\python.exe -c "import sqlite3; from pathlib import Path; p=Path(r'examples/data/project team'); c=sqlite3.connect(p/'team_selection.db'); c.executescript((p/'team_selection.sql').read_text()); c.close()"
```

