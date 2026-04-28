from xanatomy.analysis.kyphosis import severity_from_cobb
from xanatomy.analysis.scoliosis import cobb_angle_degrees, has_centroids_both_sides, severity_from_scd


def test_scoliosis_severity_bands():
    assert severity_from_scd(None) == "unknown"
    assert severity_from_scd(5) == "negative"
    assert severity_from_scd(20) == "mild"
    assert severity_from_scd(30) == "moderate"
    assert severity_from_scd(50) == "severe"


def test_kyphosis_severity_bands():
    assert severity_from_cobb(None) == "unknown"
    assert severity_from_cobb(45) == "negative"
    assert severity_from_cobb(55) == "mild"
    assert severity_from_cobb(65) == "moderate"
    assert severity_from_cobb(75) == "severe"


def test_cobb_angle_proxy_on_simple_curve():
    points = [(0.0, 0.0), (1.0, 1.0), (0.0, 2.0)]
    result = cobb_angle_degrees(points)
    assert result is not None
    angle, apex_index = result
    assert apex_index == 1
    assert angle > 0


def test_both_sides_detection():
    assert has_centroids_both_sides([(0.0, 0.0), (1.0, 1.0), (-1.0, 2.0), (0.0, 3.0)])
    assert not has_centroids_both_sides([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
