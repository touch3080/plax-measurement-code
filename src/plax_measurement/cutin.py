"""Paired binary cut-in summaries from aggregate counts only."""
from __future__ import annotations

import csv
from numbers import Integral
from pathlib import Path
import re

from scipy.stats import binomtest


def marginal(k: int, n: int) -> dict:
    """Two-sided 95% Wilson interval without continuity correction."""
    interval = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    return {
        "positive": k, "total": n, "proportion": k / n, "percent": 100 * k / n,
        "wilson_95_ci_proportion": [float(interval.low), float(interval.high)],
        "wilson_95_ci_percent": [100 * float(interval.low), 100 * float(interval.high)],
    }


def paired_binary_summary(n00: int, n01: int, n10: int, n11: int) -> dict:
    """Rows are static negative/positive; columns dynamic negative/positive.

    The rate difference is static minus dynamic. Marginal Wilson intervals must
    not be interpreted as an interval for the paired difference.
    """
    counts = (n00, n01, n10, n11)
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in counts):
        raise ValueError("All four paired cells must be nonnegative integers")
    n00, n01, n10, n11 = map(int, counts)
    n = n00 + n01 + n10 + n11
    if n == 0:
        raise ValueError("Paired table must have positive total N")
    discordant = n01 + n10
    p = float(binomtest(min(n01, n10), discordant, p=0.5, alternative="two-sided").pvalue) if discordant else 1.0
    return {
        "n": n, "row_order": ["static_negative", "static_positive"],
        "column_order": ["dynamic_negative", "dynamic_positive"],
        "paired_table": [[n00, n01], [n10, n11]],
        "cell_counts": {
            "n00_both_negative": n00, "n01_static_negative_dynamic_positive": n01,
            "n10_static_positive_dynamic_negative": n10, "n11_both_positive": n11,
        },
        "static": marginal(n10 + n11, n), "dynamic": marginal(n01 + n11, n),
        "static_minus_dynamic_percentage_points": 100 * (n10 - n01) / n,
        "paired_difference_ci": None,
        "paired_difference_ci_note": "Not estimated; marginal Wilson intervals are not paired-difference intervals.",
        "mcnemar": {
            "method": "exact two-sided binomial on discordant pairs",
            "discordant_n": discordant, "k": min(n01, n10), "null_p": 0.5,
            "p_value": p, "multiplicity_adjustment": "none; post hoc exploratory",
        },
    }


def read_aggregate_tables(path: str | Path) -> dict[str, tuple[int, int, int, int]]:
    """Read four explicit cells per analysis set; never infer omitted zeros."""
    rows = {"static_negative": 0, "static_positive": 1}
    columns = {"dynamic_negative": 0, "dynamic_positive": 1}
    grouped = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != {"analysis_set", "static_status", "dynamic_status", "n"}:
            raise ValueError("CSV requires analysis_set, static_status, dynamic_status, n")
        for record in reader:
            group = record["analysis_set"].strip()
            if not group or record["static_status"] not in rows or record["dynamic_status"] not in columns:
                raise ValueError("Every row requires a named analysis set and explicit binary status labels")
            count = record["n"].strip()
            if not re.fullmatch(r"[0-9]+", count):
                raise ValueError("Cell counts must be nonnegative integers")
            index = 2 * rows[record["static_status"]] + columns[record["dynamic_status"]]
            cells = grouped.setdefault(group, {})
            if index in cells:
                raise ValueError("Duplicate paired cell in an analysis set")
            cells[index] = int(count)
    if not grouped or any(set(cells) != {0, 1, 2, 3} for cells in grouped.values()):
        raise ValueError("Each analysis set must supply all four cells, including explicit zeros")
    return {name: tuple(cells[i] for i in range(4)) for name, cells in grouped.items()}
