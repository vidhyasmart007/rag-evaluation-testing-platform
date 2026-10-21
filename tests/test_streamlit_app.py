"""Tests for the dashboard app itself.

The app is a Streamlit script, so it is checked by running it headlessly
with Streamlit's own test harness and looking for exceptions. Streamlit is
an optional extra, so the test skips when it is not installed rather than
failing a machine that only wants to run the framework.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402 - after the skip above

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def _run() -> AppTest:
    """Run the app once and return the test object."""
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    return app


def test_the_app_opens_without_an_error():
    """A dashboard that throws on load is worse than no dashboard."""
    app = _run()

    assert [str(error.value) for error in app.exception] == []


def test_the_app_shows_the_four_views():
    """A reader can reach the numbers, the cases, the judge and the notes."""
    app = _run()

    assert [tab.label for tab in app.tabs] == [
        "Overview",
        "Cases",
        "Judge validation",
        "About",
    ]


def test_the_app_lists_the_reports_that_are_on_disk():
    """The run picker is built from the reports, not from a hard-coded list."""
    app = _run()

    options = app.sidebar.selectbox[0].options

    assert options
    assert any("baseline" in option for option in options)


def test_the_headline_numbers_are_shown_for_the_selected_run():
    """A metric card is drawn for each headline metric that has a value."""
    app = _run()

    labels = [metric.label for metric in app.metric]

    assert "Answer correct" in labels
    assert "Token F1" in labels
    assert "Faithfulness" in labels


def test_switching_to_another_run_changes_the_numbers():
    """The picker is wired to the report, not decorative."""
    app = _run()
    before = dict((m.label, m.value) for m in app.metric)

    options = app.sidebar.selectbox[0].options
    judged = next((o for o in options if "phase4_judge_lexical" in o), None)
    if judged is None:
        pytest.skip("The judged run is not in this checkout.")

    app.sidebar.selectbox[0].set_value(judged)
    app.run()
    after = dict((m.label, m.value) for m in app.metric)

    assert after["Judge correct"] != before["Judge correct"]


def test_the_judge_view_reports_the_agreement_and_the_disagreements():
    """The validation numbers are on the page, not just in the report."""
    app = _run()

    captions = [caption.value for caption in app.caption]

    assert any("kappa" in caption for caption in captions)
