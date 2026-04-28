from pathlib import Path

from xanatomy.analysis.common import is_acceptable, parse_gt_severity


def test_is_acceptable_with_tolerance():
    bands = {"negative": (-1.0, 0.5), "mild": (0.5, 1.0)}
    assert is_acceptable(0.49, "negative", bands)
    assert is_acceptable(0.51, "negative", bands, tolerate=0.02)
    assert not is_acceptable(0.6, "negative", bands, tolerate=0.02)


def test_parse_gt_severity(tmp_path: Path):
    status = tmp_path / "status.txt"
    status.write_text("cardiomegaly: positive (moderate)\n", encoding="utf-8")
    assert parse_gt_severity(status, "cardiomegaly") == "moderate"
