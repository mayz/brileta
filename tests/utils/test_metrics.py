import pytest

from brileta.util.metrics import CumulativeVar, MostRecentNVar


class TestMostRecentNVar:
    def test_initial_state(self) -> None:
        var = MostRecentNVar(num_samples=10)
        assert var.sample_count == 0
        assert var.p50 == 0.0
        assert var.p95 == 0.0
        assert var.p99 == 0.0
        assert var.get_percentiles_string() == "p50=0.00 p95=0.00 p99=0.00"

    def test_record_and_get_percentiles_less_than_capacity(self) -> None:
        var = MostRecentNVar(num_samples=10)
        for i in range(5):
            var.record(i)
        assert var.sample_count == 5
        p50, p95, p99 = var.get_percentiles()
        assert p50 == pytest.approx(2.0)
        assert p95 == pytest.approx(3.8)
        assert p99 == pytest.approx(3.96)

    def test_record_and_get_percentiles_at_capacity(self) -> None:
        var = MostRecentNVar(num_samples=10)
        for i in range(10):
            var.record(i)
        assert var.sample_count == 10
        p50, p95, p99 = var.get_percentiles()
        assert p50 == pytest.approx(4.5)
        assert p95 == pytest.approx(8.55)
        assert p99 == pytest.approx(8.91)

    def test_record_and_get_percentiles_over_capacity(self) -> None:
        var = MostRecentNVar(num_samples=10)
        for i in range(15):
            var.record(i)
        assert var.sample_count == 10
        # The samples should be [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        p50, p95, p99 = var.get_percentiles()
        assert p50 == pytest.approx(9.5)
        assert p95 == pytest.approx(13.55)
        assert p99 == pytest.approx(13.91)

    def test_get_percentiles_string(self) -> None:
        var = MostRecentNVar(num_samples=10)
        for i in range(10):
            var.record(i)
        assert var.get_percentiles_string() == "p50=4.50 p95=8.55 p99=8.91"


class TestCumulativeVar:
    def test_initial_state(self) -> None:
        var = CumulativeVar(num_samples=10)
        assert var.sample_count == 0
        assert var.p50 == 0.0
        assert var.p95 == 0.0
        assert var.p99 == 0.0
        assert var.get_percentiles_string() == "p50=0.00 p95=0.00 p99=0.00"

    def test_record_and_get_percentiles_less_than_capacity(self) -> None:
        var = CumulativeVar(num_samples=10)
        for i in range(5):
            var.record(i)
        assert var.sample_count == 5
        p50, p95, p99 = var.get_percentiles()
        assert p50 == pytest.approx(2.0)
        assert p95 == pytest.approx(3.8)
        assert p99 == pytest.approx(3.96)

    def test_record_and_get_percentiles_at_capacity(self) -> None:
        var = CumulativeVar(num_samples=10)
        for i in range(10):
            var.record(i)
        assert var.sample_count == 10
        p50, p95, p99 = var.get_percentiles()
        assert p50 == pytest.approx(4.5)
        assert p95 == pytest.approx(8.55)
        assert p99 == pytest.approx(8.91)

    def test_record_over_capacity_reservoir_sampling(self) -> None:
        var = CumulativeVar(num_samples=100)
        for i in range(200):
            var.record(i)
        assert var.sample_count == 200
        # Due to reservoir sampling, we can't know the exact values,
        # but we can check if the percentiles are within a reasonable range.
        p50, p95, p99 = var.get_percentiles()
        assert 0 < p50 < 200
        assert 0 < p95 < 200
        assert 0 < p99 < 200

    def test_get_percentiles_string(self) -> None:
        var = CumulativeVar(num_samples=10)
        for i in range(10):
            var.record(i)
        assert var.get_percentiles_string() == "p50=4.50 p95=8.55 p99=8.91"
