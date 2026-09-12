import json
import math
from pathlib import Path
from statistics import NormalDist
import subprocess
import sys

import pytest

from plax_measurement.cutin import paired_binary_summary, read_aggregate_tables


def test_one_direction_discordance_matches_exact_tail_and_zero_event_wilson():
    result = paired_binary_summary(64, 0, 32, 0)
    assert result["mcnemar"]["p_value"] == 2 ** -31
    assert result["static_minus_dynamic_percentage_points"] == pytest.approx(100 / 3)
    assert result["paired_difference_ci"] is None
    z = NormalDist().inv_cdf(0.975)
    assert result["dynamic"]["wilson_95_ci_proportion"] == pytest.approx([0, z*z/(96+z*z)])


def test_reversing_paths_reverses_difference_and_preserves_paired_p():
    original = paired_binary_summary(11, 3, 17, 9)
    reversed_paths = paired_binary_summary(11, 17, 3, 9)
    expected = min(1.0, 2 * sum(math.comb(20, j) for j in range(4)) / 2**20)
    assert original["mcnemar"]["p_value"] == pytest.approx(expected)
    assert reversed_paths["mcnemar"]["p_value"] == original["mcnemar"]["p_value"]
    assert reversed_paths["static_minus_dynamic_percentage_points"] == -original["static_minus_dynamic_percentage_points"]
    assert original["static"] == reversed_paths["dynamic"]


def test_concordant_pairs_have_no_discordance():
    result = paired_binary_summary(8, 0, 0, 4)
    assert result["mcnemar"]["p_value"] == 1
    assert result["mcnemar"]["discordant_n"] == 0
    assert result["static_minus_dynamic_percentage_points"] == 0


@pytest.mark.parametrize("counts", [(0, 0, 0, 0), (2, -1, 0, 1), (2, 1.5, 0, 1), (2, True, 0, 1)])
def test_invalid_counts_are_not_coerced(counts):
    with pytest.raises(ValueError):
        paired_binary_summary(*counts)


def table_csv():
    return ("analysis_set,static_status,dynamic_status,n\n"
            "synthetic,static_negative,dynamic_negative,7\n"
            "synthetic,static_negative,dynamic_positive,1\n"
            "synthetic,static_positive,dynamic_negative,3\n"
            "synthetic,static_positive,dynamic_positive,2\n")


@pytest.mark.parametrize("invalid", [lambda text: text.rsplit("synthetic", 1)[0], lambda text: text + text.splitlines()[1] + "\n"])
def test_missing_or_duplicate_cells_are_rejected(tmp_path, invalid):
    path = tmp_path / "cells.csv"
    path.write_text(invalid(table_csv()))
    with pytest.raises(ValueError):
        read_aggregate_tables(path)


def test_portable_cli_accepts_only_aggregate_counts(tmp_path):
    path = tmp_path / "cells.csv"
    path.write_text(table_csv())
    before = path.read_bytes()
    out = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "scripts/analyze_paired_cutin.py"
    subprocess.run([sys.executable, str(script), "--input", str(path), "--output-dir", str(out)], check=True, capture_output=True, text=True)
    result = json.loads((out / "paired_cutin_recomputed.json").read_text())
    assert result["sets"]["synthetic"]["paired_table"] == [[7, 1], [3, 2]]
    assert result["sets"]["synthetic"]["n"] == 13
    assert result["published_study_context"]["static_path_named_model"] is None
    assert path.read_bytes() == before
