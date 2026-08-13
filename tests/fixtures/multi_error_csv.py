"""Real multi-error CSV fixture content, shared across tests and the real
CLI demo, so what's tested and what's demoed are genuinely the same file.

Content only — writing to a real path (`tmp_path` in unit tests, an
explicit path for the live demo) stays with each caller, matching this
project's established convention of never committing raw sample CSV
bytes into the repo, only the Python source that deterministically
produces them.

Empirically verified (see the multi-error detection ticket's
investigation) against the real `chardet`/`csv.Sniffer` machinery:

- `TWO_FAILURE_TEXT` (Windows-1252-encoded): undecodable as UTF-8
  (-> WRONG_ENCODING) and confidently sniffed as semicolon-delimited
  (-> WRONG_DELIMITER), with otherwise clean, consistent structure — no
  third failure is present, by construction (see module docstring of
  `local_csv_failure_detector.py` for why a genuine three-way
  combination including WRONG_DELIMITER is not achievable with the
  current `csv.Sniffer`-based mechanism).

- `ENCODING_AND_SINGLE_COLUMN_TEXT`: undecodable as UTF-8
  (-> WRONG_ENCODING) and, once decoded, whitespace-separated with no
  delimiter `csv.Sniffer` can identify (-> SINGLE_COLUMN_MALFORMATION) —
  a second, independently-verified two-way combination.
"""

TWO_FAILURE_TEXT = "id;name;age\n1;café;30\n2;naïve;25\n3;façade;35\n"
TWO_FAILURE_ENCODING = "latin-1"

ENCODING_AND_SINGLE_COLUMN_TEXT = "id name age\n1 café 30\n2 naïve 25\n3 façade 35\n"
ENCODING_AND_SINGLE_COLUMN_ENCODING = "latin-1"


def two_failure_bytes() -> bytes:
    """WRONG_ENCODING + WRONG_DELIMITER, cleanly repairable via one
    combined `CsvRepairParams` (`encoding` + `delimiter`)."""
    return TWO_FAILURE_TEXT.encode(TWO_FAILURE_ENCODING)


def encoding_and_single_column_bytes() -> bytes:
    """WRONG_ENCODING + SINGLE_COLUMN_MALFORMATION."""
    return ENCODING_AND_SINGLE_COLUMN_TEXT.encode(ENCODING_AND_SINGLE_COLUMN_ENCODING)
