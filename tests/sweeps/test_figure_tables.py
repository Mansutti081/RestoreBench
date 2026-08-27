# ABOUTME: Tests the distribution helper behind the box-plot figures in build_figure_tables.
# ABOUTME: Guards the rule that a missing measurement is dropped, never read as zero.
from __future__ import annotations

from restorebench.sweeps import build_figure_tables as tables


def _row(model: str, arch: str, **columns: str) -> dict[str, str]:
    return {"model_display": model, "architecture": arch, **columns}


def test_measured_models_keep_the_canonical_order():
    # Arrange: rows arrive in whatever order the store globbed them in.
    rows = [_row("GLM-5", "chatbot"), _row("Claude Opus 5", "chatbot"), _row("Kimi K2.5", "chatbot")]

    # Act
    measured = tables.measured_models(rows)

    # Assert
    assert measured == ["Claude Opus 5", "Kimi K2.5", "GLM-5"]


def test_a_model_with_no_measured_row_counts_as_pending():
    # A hardcoded pending list keeps printing NaN after the campaign has run; deriving it from
    # the data means the file stops lying the moment the cells land.
    rows = [_row("Claude Opus 5", "chatbot")]

    assert "DeepSeek V3.2" in tables.pending_models(rows)
    assert "Claude Opus 5" not in tables.pending_models(rows)


def test_nothing_is_pending_once_every_model_is_measured():
    rows = [_row(model, "chatbot") for model in tables.ALL_MODELS]

    assert tables.pending_models(rows) == []


def test_an_unknown_model_is_still_reported():
    # A model absent from ALL_MODELS would otherwise vanish from every figure in silence.
    rows = [_row("Some New Model", "chatbot")]

    assert "Some New Model" in tables.measured_models(rows)


def test_distribution_summarises_only_the_requested_combination():
    # Arrange
    rows = [
        _row("Haiku", "chatbot", execution_time_s="10"),
        _row("Haiku", "chatbot", execution_time_s="20"),
        _row("Haiku", "chatbot", execution_time_s="30"),
        _row("Haiku", "multi-agent", execution_time_s="900"),
        _row("Opus", "chatbot", execution_time_s="900"),
    ]

    # Act
    summary = tables._distribution(rows, "Haiku", "chatbot", "execution_time_s")

    # Assert
    assert summary["n"] == 3
    assert summary["median"] == 20
    assert summary["min"] == 10
    assert summary["max"] == 30
    assert summary["mean"] == 20


def test_distribution_drops_missing_values_instead_of_counting_them_as_zero():
    # Arrange
    rows = [
        _row("Haiku", "chatbot", buses_in_band_pct="80"),
        _row("Haiku", "chatbot", buses_in_band_pct=""),
        _row("Haiku", "chatbot", buses_in_band_pct="None"),
        _row("Haiku", "chatbot", buses_in_band_pct="90"),
    ]

    # Act
    summary = tables._distribution(rows, "Haiku", "chatbot", "buses_in_band_pct")

    # Assert
    assert summary["n"] == 2
    assert summary["min"] == 80
    assert summary["median"] == 85


def test_distribution_of_an_unmeasured_combination_is_empty_not_zero():
    # Arrange
    rows = [_row("Haiku", "chatbot", execution_time_s="10")]

    # Act
    summary = tables._distribution(rows, "Kimi K2.5", "multi-agent", "execution_time_s")

    # Assert
    assert summary["n"] == 0
    assert summary["median"] is None
    assert summary["mean"] is None
    assert tables._f(summary["median"]) == "NaN"


def test_distribution_keeps_unsolved_cells_when_the_column_is_defined_for_them():
    # Arrange — unlike a final voltage profile, a runtime exists whether or not the case was solved.
    rows = [
        _row("Haiku", "chatbot", execution_time_s="100", solved="1"),
        _row("Haiku", "chatbot", execution_time_s="300", solved="0"),
    ]

    # Act
    summary = tables._distribution(rows, "Haiku", "chatbot", "execution_time_s")

    # Assert
    assert summary["n"] == 2
    assert summary["median"] == 200
