"""The dashboard.

This file is deliberately thin. Every number it shows is read from a report
that already exists on disk, by :mod:`rageval.dashboard`, so the app needs
no model server and the deployed copy shows exactly what the run produced.

Run it locally with:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# The package lives in src/ and is not installed in the deployed copy, so
# the path is added here rather than relying on an install step that the
# public deployment might not do.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402 - the path is set up above

from rageval import dashboard  # noqa: E402 - the path is set up above
from rageval.dashboard import (  # noqa: E402 - the path is set up above
    HEADLINE_METRICS,
    OTHER_METRICS,
    RunSummary,
)

st.set_page_config(
    page_title="RAG evaluation platform",
    page_icon="📊",
    layout="wide",
)

TAB_OVERVIEW = "Overview"
TAB_CASES = "Cases"
TAB_JUDGE = "Judge validation"
TAB_ABOUT = "About"


@st.cache_data(show_spinner=False)
def _runs() -> list[RunSummary]:
    """The reports on disk, read once per session."""
    return dashboard.discover_runs()


@st.cache_data(show_spinner=False)
def _run(path: str) -> dict:
    """One report, read once per session."""
    return dashboard.load_run(Path(path))


@st.cache_data(show_spinner=False)
def _validation() -> dict | None:
    """The judge validation report, if there is one."""
    return dashboard.load_judge_validation()


def _metric_selectbox(label: str, runs: list[RunSummary], default: str) -> str:
    """A metric chooser that only offers metrics at least one run has."""
    available = sorted({key for run in runs for key in run.metrics})
    if default not in available and available:
        default = available[0]
    return st.selectbox(label, available, index=available.index(default))


def _render_overview(runs: list[RunSummary], selected: RunSummary) -> None:
    """The headline numbers for one run, and a comparison across runs."""
    report = _run(str(selected.path))
    aggregates = report.get("aggregates", {}) or {}

    st.subheader(selected.run_id)
    st.caption(selected.label)

    columns = st.columns(5)
    for index, metric in enumerate(HEADLINE_METRICS):
        with columns[index % 5]:
            value = aggregates.get(metric.key)
            st.metric(
                metric.label,
                dashboard.format_metric(value, metric.kind),
                help=metric.description,
            )

    st.markdown("#### How this run is configured")
    detail_columns = st.columns(4)
    for index, (name, value) in enumerate(dashboard.run_details(report)):
        with detail_columns[index % 4]:
            st.markdown(f"**{name}**")
            st.write(value)

    st.markdown("#### Every metric in this report")
    st.caption("The numbers the run recorded, with what each one means.")
    known = {metric.key: metric.description for metric in HEADLINE_METRICS}
    table = []
    for key, value in sorted(aggregates.items()):
        if not isinstance(value, (int, float)):
            continue
        table.append(
            {
                "Metric": key,
                "Value": dashboard.format_metric(value),
                "What it means": known.get(key, OTHER_METRICS.get(key, "")),
            }
        )
    st.dataframe(table, width="stretch", hide_index=True)

    st.markdown("#### The same metric across every run")
    key = _metric_selectbox("Metric to compare", runs, "answer_correct")
    comparison = dashboard.metric_comparison(runs, key)
    if comparison:
        st.bar_chart(comparison)
    else:
        st.info(f"No run recorded {key}.")


def _render_cases(selected: RunSummary) -> None:
    """Every case in the run, and the evidence behind its score."""
    report = _run(str(selected.path))
    rows = dashboard.case_rows(report)

    left, middle, right = st.columns(3)
    with left:
        only_misses = st.checkbox(
            "Only show disagreements",
            value=False,
            help="Cases where the judge and the token rule reached different verdicts.",
        )
    with middle:
        answerable = st.selectbox(
            "Answerable", ["All", "Answerable", "Unanswerable"], index=0
        )
    with right:
        judge_filter = st.selectbox(
            "Judge verdict", ["All", "Correct", "Incorrect", "Not judged"], index=0
        )

    judged = any(row["judge_correct"] is not None for row in rows)
    if only_misses and not judged:
        st.info(
            "This run has no judge, so there are no disagreements to show. "
            "Pick a run that was judged, such as `phase4_judge_lexical`."
        )

    shown = []
    for row in rows:
        if answerable == "Answerable" and not row["answerable"]:
            continue
        if answerable == "Unanswerable" and row["answerable"]:
            continue
        if judge_filter == "Correct" and row["judge_correct"] is not True:
            continue
        if judge_filter == "Incorrect" and row["judge_correct"] is not False:
            continue
        if judge_filter == "Not judged" and row["judge_correct"] is not None:
            continue
        if only_misses and judged:
            token = row["token_correct"]
            verdict = row["judge_correct"]
            if token is not None and verdict is not None and token == verdict:
                continue
        shown.append(row)

    st.caption(f"{len(shown)} of {len(rows)} cases.")
    if not shown:
        st.info("No case matches those filters.")
        return

    st.dataframe(
        [
            {
                "Case": row["id"],
                "Question": row["question"],
                "Answer": row["answer"],
                "Token rule": dashboard.judge_verdict_label(row["token_correct"]),
                "Judge": dashboard.judge_verdict_label(row["judge_correct"]),
                "F1": dashboard.format_metric(row["token_f1"]),
            }
            for row in shown
        ],
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Case by case")
    for row in shown:
        title = f"Case {row['id']} — {row['question']}"
        with st.expander(title):
            columns = st.columns(2)
            with columns[0]:
                st.markdown("**The answer**")
                st.write(row["answer"] or "_(the pipeline declined to answer)_")
                st.markdown("**The reference**")
                st.write(row["reference"] or "_(there is none)_")
            with columns[1]:
                st.markdown("**Verdicts**")
                st.write(
                    {
                        "Token rule": dashboard.judge_verdict_label(
                            row["token_correct"]
                        ),
                        "Judge": dashboard.judge_verdict_label(row["judge_correct"]),
                        "Token F1": dashboard.format_metric(row["token_f1"]),
                        "Faithfulness": dashboard.format_metric(row["faithfulness"]),
                    }
                )
            if row["judge_reason"]:
                st.markdown("**Why the judge said so**")
                st.write(row["judge_reason"])
            if row["chunks"]:
                st.markdown("**What was retrieved**")
                for chunk in row["chunks"]:
                    st.markdown(
                        f"`{chunk['chunk_id']}` · score {chunk['score']:.4f} · "
                        f"sentences {chunk['sentence_ids']}"
                    )
                    st.write(chunk["text"])
            if row["error"]:
                st.error(row["error"])


def _render_judge_validation() -> None:
    """How well the judge was measured against the hand labels."""
    report = _validation()
    if report is None:
        st.info(
            "No judge validation report was found. Run "
            "`make validate-judge` to produce one."
        )
        return

    consistency = dashboard.self_consistency(report)
    columns = st.columns(4)
    with columns[0]:
        st.metric(
            "Sample agreement",
            dashboard.format_metric(consistency["sample_agreement"], "percent"),
            help="Whether the judge's own samples reached the same verdict.",
        )
    with columns[1]:
        st.metric(
            "Order consistency",
            dashboard.format_metric(consistency["order_consistency"], "percent"),
            help="Whether it reached the same verdict with the order swapped.",
        )
    with columns[2]:
        st.metric(
            "Score kappa",
            dashboard.format_metric(consistency["score_kappa"]),
            help="Agreement on the 1 to 5 scores, not just the verdict.",
        )
    with columns[3]:
        st.metric(
            "Items",
            str(report.get("n_items", 0)),
            help="Hand labelled answers the judge was measured against.",
        )

    st.markdown("#### Agreement with the hand labels")
    st.dataframe(
        dashboard.kappa_rows(report),
        width="stretch",
        hide_index=True,
    )

    against = dashboard.judge_vs_token_row(report)
    if against:
        st.caption(
            f"The judge and the token rule agree with each other on "
            f"{dashboard.format_metric(against['agreement'], 'percent')} of the "
            f"{against['n_items']} items (kappa "
            f"{dashboard.format_metric(against['kappa'])})."
        )

    st.markdown("#### Mean rubric scores")
    scores = dashboard.mean_judge_scores(report)
    if scores:
        st.bar_chart(scores)
        st.caption(
            f"Judge: {consistency.get('judge_name', 'local')} · "
            f"{consistency.get('n_samples', 3)} samples per order · "
            f"threshold {consistency.get('threshold', 4)}."
        )

    st.markdown("#### Where a rater disagreed with the labels")
    disagreements = dashboard.judge_disagreements(report)
    if disagreements:
        st.dataframe(
            [
                {
                    "Case": row["test_id"],
                    "Question": row["question"],
                    "Answer": row["answer"],
                    "Hand label": row["human"],
                    "Judge": row["judge"],
                    "Token rule": row["token_rule"],
                    "Tuned on": "yes" if row["used_for_tuning"] else "",
                    "Note": row["note"],
                }
                for row in disagreements
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.success("Every rater agreed with every hand label.")


def _render_about() -> None:
    """What this is, and what it is not."""
    st.markdown(
        """
### What this shows

Every number here was produced by a run of the framework and saved as a
JSON report. The dashboard reads those reports and nothing else. It does
not call a model, so what a reader sees is exactly what the run recorded,
and the deployed copy behaves the same as the local one.

### How it was produced

* **Retrieval** is BM25 over the knowledge base, with a dense arm using
  Ollama embeddings and a hybrid arm that blends the two.
* **Generation** is a local model through Ollama, seeded, so the same
  configuration produces the same answers.
* **Judging** is a second model from a different family, asked for four
  rubric scores and nothing else. The pass or fail verdict is derived from
  the correctness score, so the report cannot carry two fields that
  contradict each other.

### What it is not

* It is **not** a live demo. No model runs when you open this page; the
  reports are the ones committed to the repository.
* The judge sees only the retrieved context, so it cannot detect a
  retrieval failure. That job belongs to the retrieval metrics.
* The rubric was written with part of the validation set open. The
  held-out figures are the honest measurement and are reported separately.

### Where it came from

The full method, the validation numbers and the known limitations are in
the repository's README.
"""
    )


def main() -> None:
    """Draw the page."""
    runs = _runs()
    st.title("RAG evaluation platform")

    if not runs:
        st.warning(
            "No run report was found in `reports/`. Run "
            "`python -m rageval.cli run` to produce one."
        )
        return

    labels = [f"{run.run_id} · {run.label}" for run in runs]
    chosen = st.sidebar.selectbox("Run", labels, index=0)
    selected = runs[labels.index(chosen)]
    st.sidebar.caption(
        f"{selected.n_cases} cases · {selected.created_at or 'unknown date'}"
    )

    tabs = st.tabs([TAB_OVERVIEW, TAB_CASES, TAB_JUDGE, TAB_ABOUT])
    with tabs[0]:
        _render_overview(runs, selected)
    with tabs[1]:
        _render_cases(selected)
    with tabs[2]:
        _render_judge_validation()
    with tabs[3]:
        _render_about()


main()
