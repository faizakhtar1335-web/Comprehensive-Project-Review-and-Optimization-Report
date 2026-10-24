"""
Week 4: Comprehensive Project Review and Optimization
-----------------------------------------------------
A reusable EDA and data-quality review script.

Usage:
    python week4_project_review.py --input path/to/data.csv --output-dir outputs

Core dependencies:
    python -m pip install pandas numpy matplotlib seaborn

The script is dataset-agnostic. It profiles a CSV, performs conservative cleaning,
creates example visualizations for numeric columns, and writes a text summary.
It deliberately does not delete outliers automatically or invent business findings.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Return a compact, JSON-serializable profile of the input dataframe."""
    missing = df.isna().sum()
    profile = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [str(c) for c in df.columns],
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "missing_count": {str(k): int(v) for k, v in missing.items()},
        "missing_percent": {
            str(k): round(float(v / max(len(df), 1) * 100), 2)
            for k, v in missing.items()
        },
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": {},
    }

    numeric = df.select_dtypes(include=np.number)
    if not numeric.empty:
        profile["numeric_summary"] = (
            numeric.describe()
            .replace({np.nan: None})
            .round(4)
            .to_dict()
        )
    return profile


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply conservative, transparent cleaning.

    - Trims whitespace from column names and string cells.
    - Removes exact duplicate rows only.
    - Does not impute missing values or automatically remove outliers.
    """
    cleaned = df.copy()
    before_rows = len(cleaned)

    cleaned.columns = [str(col).strip() for col in cleaned.columns]

    object_cols = cleaned.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        cleaned[col] = cleaned[col].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    duplicate_count = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().copy()

    log = {
        "rows_before": int(before_rows),
        "rows_after": int(len(cleaned)),
        "exact_duplicates_removed": duplicate_count,
        "rows_removed_total": int(before_rows - len(cleaned)),
        "notes": [
            "Only exact duplicate rows were removed.",
            "Missing values were retained for explicit review.",
            "Outliers were not automatically deleted.",
            "Confirm that exact duplicates are invalid for your dataset before using this step.",
        ],
    }
    return cleaned, log


def save_missingness_plot(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Save a bar chart of missing-value percentages by column, if applicable."""
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    if missing_pct.empty:
        LOGGER.info("No missing values detected; skipping missingness chart.")
        return None

    fig_height = max(4, min(10, 0.35 * len(missing_pct)))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    sns.barplot(x=missing_pct.values, y=missing_pct.index.astype(str), ax=ax)
    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("Column")
    fig.tight_layout()
    path = output_dir / "missing_values.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_numeric_plots(df: pd.DataFrame, output_dir: Path, max_columns: int = 6) -> list[str]:
    """Save histograms and a correlation heatmap for numeric columns."""
    numeric = df.select_dtypes(include=np.number)
    saved: list[str] = []

    for col in list(numeric.columns)[:max_columns]:
        values = numeric[col].dropna()
        if values.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(values, kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(str(col))
        ax.set_ylabel("Count")
        fig.tight_layout()
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(col))
        path = output_dir / f"distribution_{safe_name}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))

    if numeric.shape[1] >= 2:
        corr = numeric.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(corr.columns)), max(5, 0.7 * len(corr.columns))))
        sns.heatmap(corr, cmap="vlag", center=0, annot=(len(corr.columns) <= 10), fmt=".2f", ax=ax)
        ax.set_title("Correlation Matrix (numeric columns)")
        fig.tight_layout()
        path = output_dir / "correlation_heatmap.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))

    return saved


def write_review_summary(
    raw_profile: dict[str, Any],
    cleaned_profile: dict[str, Any],
    cleaning_log: dict[str, Any],
    runtime_seconds: float,
    output_dir: Path,
) -> Path:
    """Write a human-readable summary with measured profile information."""
    lines = [
        "COMPREHENSIVE PROJECT REVIEW — AUTOMATED DATA PROFILE",
        "=" * 58,
        "",
        "IMPORTANT: This summary reports dataset diagnostics, not causal conclusions.",
        "",
        "INPUT PROFILE",
        f"- Rows: {raw_profile['rows']}",
        f"- Columns: {raw_profile['columns']}",
        f"- Exact duplicate rows: {raw_profile['exact_duplicate_rows']}",
        "",
        "CLEANING LOG",
        f"- Rows before: {cleaning_log['rows_before']}",
        f"- Rows after: {cleaning_log['rows_after']}",
        f"- Exact duplicates removed: {cleaning_log['exact_duplicates_removed']}",
        f"- Runtime (seconds): {runtime_seconds:.3f}",
        "",
        "MISSING VALUES (%) BEFORE CLEANING",
    ]
    for col, pct in raw_profile["missing_percent"].items():
        lines.append(f"- {col}: {pct}%")

    lines.extend(["", "MISSING VALUES (%) AFTER CLEANING"])
    for col, pct in cleaned_profile["missing_percent"].items():
        lines.append(f"- {col}: {pct}%")

    lines.extend([
        "",
        "REVIEW QUESTIONS",
        "1. Are missing values concentrated in particular groups or columns?",
        "2. Are exact duplicate rows truly invalid for this dataset?",
        "3. Do numeric ranges violate documented business or domain rules?",
        "4. Are outliers valid observations, data errors, or events requiring context?",
        "5. Do chart titles, axes, units, time windows, and aggregation methods make sense?",
        "6. Have conclusions been limited to what the data can support?",
        "",
        "RECOMMENDATIONS",
        "- Review missingness before deciding whether to drop or impute values.",
        "- Validate data types, keys, ranges, units, and category labels.",
        "- Do not remove outliers without evidence and sensitivity analysis.",
        "- Benchmark alternatives only after confirming that results remain equivalent.",
        "- Re-run the script from the raw source and preserve the cleaning log.",
        "",
    ])
    path = output_dir / "project_review_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile, clean, and visualize a CSV dataset.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output-dir", default="week4_outputs", help="Directory for reports and charts.")
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="CSV encoding (default: utf-8). Use an appropriate alternative if needed.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    start = time.perf_counter()
    LOGGER.info("Loading CSV: %s", input_path)
    raw_df = pd.read_csv(input_path, encoding=args.encoding)

    raw_profile = profile_dataframe(raw_df)
    cleaned_df, cleaning_log = clean_dataframe(raw_df)
    cleaned_profile = profile_dataframe(cleaned_df)
    elapsed = time.perf_counter() - start

    # Save auditable outputs.
    (output_dir / "raw_profile.json").write_text(
        json.dumps(raw_profile, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (output_dir / "cleaned_profile.json").write_text(
        json.dumps(cleaned_profile, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (output_dir / "cleaning_log.json").write_text(
        json.dumps(cleaning_log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    cleaned_path = output_dir / "cleaned_data.csv"
    cleaned_df.to_csv(cleaned_path, index=False, encoding="utf-8")

    missing_plot = save_missingness_plot(raw_df, output_dir)
    numeric_plots = save_numeric_plots(cleaned_df, output_dir)
    summary_path = write_review_summary(
        raw_profile, cleaned_profile, cleaning_log, elapsed, output_dir
    )

    LOGGER.info("Finished.")
    LOGGER.info("Cleaned dataset: %s", cleaned_path)
    LOGGER.info("Review summary: %s", summary_path)
    if missing_plot:
        LOGGER.info("Missingness chart: %s", missing_plot)
    for plot_path in numeric_plots:
        LOGGER.info("Chart: %s", plot_path)
    LOGGER.info("Outputs saved to: %s", output_dir.resolve())


if __name__ == "__main__":
    main()
