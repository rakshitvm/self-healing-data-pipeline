# CSV Self-Healing Test Kit

This kit contains 13 CSV files designed to verify a self-healing CSV pipeline
implementation. Each file triggers a specific failure mode (or combination);
one is a control that should load cleanly with no repair.

## Test matrix

| # | File | Trigger | Correct repair | Expected result |
|---|---|---|---|---|
| 00 | `00_control_clean.csv` | none | no repair needed | 10 rows × 11 cols, no LLM call |
| 01 | `01_wrong_delimiter_pipe.csv` | pipe delimiter | `sep="\|"` | 10 rows × 11 cols |
| 02 | `02_wrong_delimiter_semicolon.csv` | semicolon delimiter | `sep=";"` | 10 rows × 11 cols |
| 03 | `03_wrong_delimiter_tab.csv` | tab delimiter | `sep="\t"` | 10 rows × 11 cols |
| 04 | `04_wrong_encoding_latin1.csv` | Latin-1 with £, é, ñ | `encoding="latin-1"` | 10 rows × 11 cols, no mojibake |
| 05 | `05_wrong_encoding_cp1252.csv` | Windows-1252 encoding | `encoding="cp1252"` | 10 rows × 11 cols |
| 06 | `06_wrong_encoding_utf16.csv` | UTF-16 encoded | `encoding="utf-16"` | 10 rows × 11 cols |
| 07 | `07_header_row_2.csv` | 1 preamble line before header | `skiprows=1` | 10 rows × 11 cols |
| 08 | `08_header_row_5.csv` | 4 preamble lines before header | `skiprows=4` | 10 rows × 11 cols |
| 09 | `09_no_header.csv` | no header row at all | `header=None` | 10 rows × 11 cols, integer col names |
| 10 | `10_bom_prefix.csv` | BOM + zero-width space in first col | `encoding="utf-8-sig"` (+ strip zero-width) | 10 rows × 11 cols, first col = `store_id` |
| 11 | `11_combined_pipe_and_latin1.csv` | pipe + Latin-1 | `sep="\|"`, `encoding="latin-1"` | 10 rows × 11 cols |
| 12 | `12_combined_semicolon_and_header.csv` | semicolon + 3-line preamble | `sep=";"`, `skiprows=3` | 10 rows × 11 cols |

## What every successful load must look like

- Row count = **10** (nothing dropped, no rows split by embedded delimiters)
- Column count = **11**
- Column names contain no invisible characters (`\ufeff`, `\u200b`)
- Values like `Café Nero`, `Piñata Chocolate 100g`, `£2.79` render correctly
  with no `Ã©`, `Ã±`, or `Â£` mojibake
- Numeric columns (`units_sold`, `retail_price`, `gross_sales`) are parseable

## Failure modes to watch for during review

**Silent single-column parse**: pandas returns `df.shape[1] == 1` when the
wrong delimiter is used but no exception fires. The implementation must
detect this explicitly (see files 01, 02, 03, 07, 12).

**Mojibake with no error**: reading Latin-1 as UTF-8 with `errors="replace"`
will not raise but produces corrupted strings. Check `store_name` and
`product_name` for garbled characters (files 04, 05).

**BOM as part of column name**: `df.columns[0]` should be exactly `"store_id"`,
not `"\ufeffstore_id"` or `"\u200bstore_id"`. File 10 was constructed with
both BOM and zero-width space — a robust implementation strips both after
reading, or uses `encoding="utf-8-sig"` and then re-strips leftover
zero-width characters.

**Header-detection distinction**: files 07 and 08 use `skiprows`, not
`header=N`. Using `header=1` on file 07 works (pandas treats index 1 as
the header), but using `header=4` on file 08 fails because pandas ignores
blank lines during header detection and picks the wrong row. Accept either
`skiprows` or a correctly-computed `header=N` — but the implementation
must land on 10 rows, not 9.

**Cost blowup**: no LLM call should fire for file 00. If the implementation
calls Azure OpenAI on a clean file, the healthy-path short-circuit is missing.

## Suggested review order

1. Run file 00 first — confirm no LLM call, no repair, clean load.
2. Run files 01-03 (delimiter) — verify the `sep` parameter is prescribed.
3. Run files 04-06 (encoding) — verify special characters survive intact.
4. Run files 07-09 (header) — verify column names are the real headers.
5. Run file 10 (BOM + ZWSP) — verify no invisible chars in `df.columns[0]`.
6. Run files 11-12 (combined) — verify **both** fixes are applied (either
   in one LLM call or via successive retries — both are acceptable).

## Reviewer checklist per file

For each file, confirm:

- [ ] Correct DataFrame shape: (10, 11)
- [ ] `list(df.columns)` starts with `['store_id', 'store_name', 'region', ...]`
- [ ] `df['store_name'].iloc[3]` returns `"Waitrose - Café Nero"` exactly
- [ ] `df['product_name'].iloc[4]` returns `"Piñata Chocolate 100g"` exactly
- [ ] `df['retail_price'].iloc[9]` contains `£` (not `Â£` or `?`)
- [ ] The intern's log clearly shows: which failure was detected, which
      params were prescribed, and how many retry attempts were used

## How to load them

```python
import os
from pipeline import SelfHealingLoader

loader = SelfHealingLoader()
kit = "csv_test_kit"

for fname in sorted(os.listdir(kit)):
    if not fname.endswith(".csv"):
        continue
    print(f"\n=== {fname} ===")
    df = loader.load(f"{kit}/{fname}")
    assert len(df) == 10, f"Expected 10 rows, got {len(df)}"
    assert df.shape[1] == 11, f"Expected 11 cols, got {df.shape[1]}"
    print(df.head(2))
```

## Assessment rubric

| Score | Criteria |
|---|---|
| **Pass** | All 13 files load with correct shape, values, and no mojibake. No LLM call on file 00. Combined-failure files (11, 12) recover in ≤3 attempts. |
| **Partial** | 10-12 files pass. Some combined failures require manual intervention. Silent-failure detection missing but exception-based failures all healed. |
| **Fail** | Fewer than 10 files pass, OR file 00 triggers an LLM call, OR mojibake present in recovered files, OR the implementation crashes on any file. |
