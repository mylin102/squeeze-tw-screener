from __future__ import annotations

from typing import Dict, Any, List

import pandas as pd
import numpy as np

from squeeze.report.performance import normalize_tracking_df

# Feature columns used for True/False comparison
_BOOLEAN_FEATURES = [
    ("has_squeeze", "Squeeze"),
    ("has_houyi", "Houyi"),
    ("has_whale", "Whale"),
    ("close_above_ma20", "Close>MA20"),
    ("ma20_converging", "MA20 Converging"),
    ("volume_expansion", "Vol Expansion"),
    ("rs_new_high_60", "RS New High 60"),
    ("hit_5pct_14d", "Hit 5% 14d"),
]


def load_tracking_frame(csv_path: str) -> pd.DataFrame:
    return normalize_tracking_df(pd.read_csv(csv_path))


def build_tracking_report(df: pd.DataFrame) -> Dict[str, Any]:
    frame = normalize_tracking_df(df)
    if frame.empty:
        return {
            "summary": {"total_records": 0, "completed_records": 0, "active_records": 0},
            "by_type": [],
            "by_signal": [],
            "by_holding_day": [],
            "by_regime": [],
            "by_score": [],
            "by_feature": [],
            "by_pattern_combo": [],
            "by_rs_quantile": [],
            "recommendations": ["No tracking data available."],
        }

    completed = frame[frame["status"] == "completed"].copy()
    active = frame[frame["status"] == "tracking"].copy()
    report: Dict[str, Any] = {
        "summary": {
            "total_records": int(len(frame)),
            "completed_records": int(len(completed)),
            "active_records": int(len(active)),
            "date_range": {"start": str(frame["date"].min()), "end": str(frame["date"].max())},
        }
    }

    if completed.empty:
        report.update({
            "by_type": [],
            "by_signal": [],
            "by_holding_day": [],
            "by_regime": [],
            "by_score": [],
            "by_feature": [],
            "by_pattern_combo": [],
            "by_rs_quantile": [],
            "recommendations": [
                "No completed records yet. Let the tracker age past 14 days before drawing conclusions.",
            ],
        })
        return report

    completed["win"] = completed["strategy_return_pct"] > 0
    completed["holding_bucket"] = completed["days_tracked"].apply(_holding_bucket)

    # Original sections
    report["by_type"] = _aggregate(completed, ["type"], rename={"type": "bucket"})
    report["by_signal"] = _aggregate(completed, ["type", "signal"], rename={"type": "direction", "signal": "bucket"})
    report["by_holding_day"] = _aggregate(completed, ["days_tracked"], rename={"days_tracked": "bucket"})
    report["by_regime"] = _aggregate(completed, ["market_regime", "type"], rename={"market_regime": "bucket", "type": "direction"})
    report["by_holding_bucket"] = _aggregate(completed, ["holding_bucket", "type"], rename={"holding_bucket": "bucket", "type": "direction"})

    # --- New validation sections ---

    # 1. Score bucketing (both v1 and v2)
    report["by_score"] = _build_score_buckets(completed)

    # 2. Feature comparison (unconditional True vs False, NA-filtered)
    report["by_feature"] = _build_feature_comparison(completed)

    # 3. Feature comparison conditional on baseline score
    report["by_feature_conditional"] = _build_feature_comparison_conditional(completed)

    # 4. Pattern combination performance
    report["by_pattern_combo"] = _build_pattern_combos(completed)

    # 5. RS quantile analysis
    report["by_rs_quantile"] = _build_rs_quantiles(completed)

    # 6. Deduplicated (first signal only per ticker)
    deduped = completed.sort_values("date").drop_duplicates(subset=["ticker"], keep="first")
    report["by_score_deduped"] = _build_score_buckets(deduped)
    report["by_feature_deduped"] = _build_feature_comparison(deduped)

    # 7. SELL shadow research features (computed in-memory only, NOT written to CSV)
    #    Four SELL-specific prev_momentum features designed for signals where
    #    prev_momentum is always negative (BUY's absolute-value buckets don't apply).
    report["sell_shadow"] = _build_sell_shadow_features(completed)

    report["recommendations"] = _derive_recommendations(report)
    return report


def format_tracking_report(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines: List[str] = [
        "Squeeze Tracking Analysis",
        f"- Total records: {summary['total_records']}",
        f"- Completed records: {summary['completed_records']}",
        f"- Active records: {summary['active_records']}",
    ]
    date_range = summary.get("date_range")
    if date_range:
        lines.append(f"- Date range: {date_range['start']} to {date_range['end']}")

    # Original sections
    lines.extend(_format_section("By Type", report.get("by_type", []), ["bucket"]))
    lines.extend(_format_section("By Signal", report.get("by_signal", []), ["direction", "bucket"]))
    lines.extend(_format_section("By Holding Day", report.get("by_holding_day", []), ["bucket"]))
    lines.extend(_format_section("By Regime", report.get("by_regime", []), ["bucket", "direction"]))
    lines.extend(_format_section("By Holding Bucket", report.get("by_holding_bucket", []), ["bucket", "direction"]))

    # --- New sections ---

    # Score bucketing
    score_rows = report.get("by_score", [])
    if score_rows:
        lines.append("")
        lines.append("Score Bucket Performance")
        lines.append(f"  {'Score':>6} {'Count':>6} {'Avg 14D':>8} {'Med 14D':>8} {'Win%':>6} {'MaxDD':>8} {'Avg 5D':>8}")
        for row in score_rows:
            lines.append(
                f"  {row['bucket']:>6} {row['sample_size']:>6} "
                f"{row['avg_14d']:>7.1f}% {row['med_14d']:>7.1f}% "
                f"{row['win_rate']:>5.1f}% {row['avg_max_dd']:>7.1f}% "
                f"{row['avg_5d']:>7.1f}%"
            )

    # Feature comparison
    feat_rows = report.get("by_feature", [])
    if feat_rows:
        lines.append("")
        lines.append("Feature Comparison (True vs False)")
        lines.append(f"  {'Feature':<20} {'n_True':>6} {'n_False':>6} {'Avg_T':>8} {'Avg_F':>8} {'Diff%':>8} {'Win_T':>6} {'Win_F':>6}")
        for row in feat_rows:
            lines.append(
                f"  {row['feature']:<20} {row['n_true']:>6} {row['n_false']:>6} "
                f"{row['avg_true']:>7.1f}% {row['avg_false']:>7.1f}% "
                f"{row['diff_pct']:>+7.1f}% {row['win_true']:>5.1f}% {row['win_false']:>5.1f}%"
            )

    # Pattern combos
    combo_rows = report.get("by_pattern_combo", [])
    if combo_rows:
        lines.append("")
        lines.append("Pattern Combination Performance")
        lines.append(f"  {'Combo':<35} {'n':>4} {'Avg 14D':>8} {'Win%':>6}")
        for row in combo_rows:
            lines.append(
                f"  {row['combo']:<35} {row['sample_size']:>4} "
                f"{row['avg_strategy_return']:>7.1f}% {row['win_rate']:>5.1f}%"
            )

    # RS quantile
    rs_rows = report.get("by_rs_quantile", [])
    if rs_rows:
        lines.append("")
        lines.append("RS Slope Quantile Performance")
        lines.append(f"  {'Quantile':<10} {'n':>4} {'Avg 14D':>8} {'Win%':>6} {'Avg RS':>8}")
        for row in rs_rows:
            lines.append(
                f"  {row['bucket']:<10} {row['sample_size']:>4} "
                f"{row['avg_strategy_return']:>7.1f}% {row['win_rate']:>5.1f}% "
                f"{row['avg_rs_slope']:>+7.4f}"
            )

    # SELL shadow research features
    shadow = report.get("sell_shadow", {})
    if shadow and shadow.get("n_sell", 0) > 0:
        lines.append("")
        lines.append(f"[Shadow Research] SELL prev_momentum Analysis (n={shadow['n_sell']}, research only — not in production)")
        lines.append(f"  {shadow.get('note', '')}")

        abs_rows = shadow.get("pm_abs_buckets", [])
        if abs_rows:
            lines.append("")
            lines.append("  A. abs(prev_momentum) × SELL win rate")
            lines.append(f"  {'abs(pm)':<10} {'n':>3} {'win%':>6} {'avg':>7} {'med':>7}")
            for r in abs_rows:
                lines.append(f"  {r['bucket']:<10} {r['n']:>3} {r['win_rate']:>6.1f} "
                             f"{r['avg_return']:>7.2f} {r['med_return']:>7.2f}")

        delta_rows = shadow.get("mom_delta_buckets", [])
        if delta_rows:
            lines.append("")
            lines.append("  B. mom_delta (momentum − prev_momentum) × SELL win rate")
            lines.append(f"  {'delta':<10} {'n':>3} {'win%':>6} {'avg':>7} {'med':>7}")
            for r in delta_rows:
                lines.append(f"  {r['bucket']:<10} {r['n']:>3} {r['win_rate']:>6.1f} "
                             f"{r['avg_return']:>7.2f} {r['med_return']:>7.2f}")

        deep_rows = shadow.get("deepening_flag", [])
        if deep_rows:
            lines.append("")
            lines.append("  C. Deepening flag (momentum < prev_momentum)")
            lines.append(f"  {'state':<12} {'n':>3} {'win%':>6} {'avg':>7} {'med':>7}")
            for r in deep_rows:
                lines.append(f"  {r['bucket']:<12} {r['n']:>3} {r['win_rate']:>6.1f} "
                             f"{r['avg_return']:>7.2f} {r['med_return']:>7.2f}")

        rd_rows = shadow.get("regime_x_deepening", [])
        if rd_rows:
            lines.append("")
            lines.append("  D. Regime × Deepening × SELL")
            lines.append(f"  {'regime':<15} {'state':<12} {'n':>3} {'win%':>6} {'avg':>7}")
            for r in rd_rows:
                lines.append(f"  {r['regime']:<15} {r['state']:<12} {r['n']:>3} "
                             f"{r['win_rate']:>6.1f} {r['avg_return']:>7.2f}")

    recommendations = report.get("recommendations", [])
    if recommendations:
        lines.append("")
        lines.append("Recommendations")
        for item in recommendations:
            lines.append(f"- {item}")
    return "\n".join(lines)



# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate(df: pd.DataFrame, group_cols: List[str], rename: Dict[str, str]) -> List[Dict[str, Any]]:
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            sample_size=("ticker", "count"),
            win_rate=("win", "mean"),
            avg_strategy_return=("strategy_return_pct", "mean"),
            median_strategy_return=("strategy_return_pct", "median"),
            avg_raw_return=("return_pct", "mean"),
        )
        .reset_index()
    )
    grouped["win_rate"] = grouped["win_rate"] * 100.0
    grouped = grouped.sort_values(by=["avg_strategy_return", "win_rate"], ascending=False)
    grouped = grouped.rename(columns=rename)
    return grouped.to_dict("records")


def _holding_bucket(days_tracked: int) -> str:
    if days_tracked <= 3:
        return "1-3d"
    if days_tracked <= 5:
        return "4-5d"
    if days_tracked <= 10:
        return "6-10d"
    return "11-14d"


# ---------------------------------------------------------------------------
# New report builders
# ---------------------------------------------------------------------------

def _build_score_buckets(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Group completed records by composite_score and composite_score_v2."""
    rows = []
    for col, label in [("composite_score", "v1"), ("composite_score_v2", "v2")]:
        if col not in df.columns or df[col].isna().all():
            continue
        grouped = (
            df.groupby(col, dropna=False)
            .agg(
                sample_size=("ticker", "count"),
                win_rate=("win", "mean"),
                avg_14d=("return_14d", "mean"),
                med_14d=("return_14d", "median"),
                avg_5d=("return_5d", "mean"),
                avg_max_dd=("max_drawdown_14d", "mean"),
            )
            .reset_index()
        )
        grouped["win_rate"] = grouped["win_rate"] * 100.0
        grouped = grouped.sort_values(col)
        for _, row in grouped.iterrows():
            rows.append({
                "bucket": f"{label}={int(row[col])}",
                "sample_size": int(row["sample_size"]),
                "win_rate": float(row["win_rate"]),
                "avg_14d": float(row["avg_14d"]),
                "med_14d": float(row["med_14d"]),
                "avg_5d": float(row["avg_5d"]),
                "avg_max_dd": float(row["avg_max_dd"]),
            })
    return rows


def _build_feature_comparison(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """For each boolean feature, compare True vs False performance."""
    rows = []
    # Use return_14d if available, otherwise strategy_return_pct
    ret_col = "return_14d" if "return_14d" in df.columns else "strategy_return_pct"
    for col, label in _BOOLEAN_FEATURES:
        if col not in df.columns:
            continue
        true_mask = df[col] == True
        false_mask = df[col] == False
        n_true = int(true_mask.sum())
        n_false = int(false_mask.sum())
        if n_true < 2 or n_false < 2:
            continue
        avg_true = float(df.loc[true_mask, ret_col].mean())
        avg_false = float(df.loc[false_mask, ret_col].mean())
        win_true = float(df.loc[true_mask, "win"].mean()) * 100.0
        win_false = float(df.loc[false_mask, "win"].mean()) * 100.0
        diff = avg_true - avg_false
        rows.append({
            "feature": label,
            "n_true": n_true,
            "n_false": n_false,
            "avg_true": avg_true,
            "avg_false": avg_false,
            "diff_pct": diff,
            "win_true": win_true,
            "win_false": win_false,
        })
    rows.sort(key=lambda r: abs(r["diff_pct"]), reverse=True)
    return rows


def _build_feature_comparison_conditional(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Feature comparison stratified by baseline ranking_score.
    Answers: within the same v1 score bucket, does RS=True outperform RS=False?
    """
    score_col = "ranking_score"
    if score_col not in df.columns:
        return []
    ret_col = "return_14d" if "return_14d" in df.columns else "strategy_return_pct"
    rows = []
    # Only compare features that have enough NA-filtered data
    for col, label in _BOOLEAN_FEATURES:
        if col not in df.columns:
            continue
        valid = df.dropna(subset=[col, score_col, ret_col])
        if len(valid) < 10:
            continue
        for score_val in sorted(valid[score_col].dropna().unique()):
            bucket = valid[valid[score_col] == score_val]
            true_b = bucket[bucket[col] == True]
            false_b = bucket[bucket[col] == False]
            if len(true_b) < 2 or len(false_b) < 2:
                continue
            avg_true = float(true_b[ret_col].mean())
            avg_false = float(false_b[ret_col].mean())
            diff = avg_true - avg_false
            rows.append({
                "feature": label,
                "baseline_score": int(score_val),
                "n_true": len(true_b),
                "n_false": len(false_b),
                "avg_true": avg_true,
                "avg_false": avg_false,
                "diff_pct": diff,
            })
    rows.sort(key=lambda r: abs(r["diff_pct"]), reverse=True)
    return rows


def _build_pattern_combos(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Group by pattern boolean combinations."""
    for col in ["has_squeeze", "has_houyi", "has_whale"]:
        if col not in df.columns:
            return []

    def _combo_label(row) -> str:
        parts = []
        if row.get("has_squeeze"):
            parts.append("Squeeze")
        if row.get("has_houyi"):
            parts.append("Houyi")
        if row.get("has_whale"):
            parts.append("Whale")
        return "+".join(parts) if parts else "None"

    df = df.copy()
    df["combo"] = df.apply(_combo_label, axis=1)
    grouped = (
        df.groupby("combo", dropna=False)
        .agg(
            sample_size=("ticker", "count"),
            win_rate=("win", "mean"),
            avg_strategy_return=("strategy_return_pct", "mean"),
            median_strategy_return=("strategy_return_pct", "median"),
        )
        .reset_index()
    )
    grouped["win_rate"] = grouped["win_rate"] * 100.0
    grouped = grouped.sort_values("avg_strategy_return", ascending=False)
    return grouped.to_dict("records")


def _build_sell_shadow_features(df: pd.DataFrame) -> dict:
    """
    Research shadow section for SELL signals only.

    Computes four SELL-specific prev_momentum features IN MEMORY — results are
    included in the tracking report text but are NEVER written back to the CSV.
    Production ranking (ranking_score, feature_schema_version) is untouched.

    Background: SELL prev_momentum is nearly always negative, so BUY's absolute-
    value quartile buckets are meaningless on the sell side.  Four alternatives:

      A. abs(prev_momentum) — negative momentum strength
      B. mom_delta = momentum − prev_momentum — acceleration direction
      C. deepening = (momentum < prev_momentum) — continuing decline flag
      D. regime × deepening cross-tab
    """
    sell = df[df["type"] == "sell"].copy()
    if sell.empty or "prev_momentum" not in sell.columns:
        return {}

    sell["prev_momentum"] = pd.to_numeric(sell["prev_momentum"], errors="coerce")
    sell["momentum"]      = pd.to_numeric(sell["momentum"],      errors="coerce")
    sell["strategy_return_pct"] = pd.to_numeric(sell["strategy_return_pct"], errors="coerce")
    sell["win"] = (sell["strategy_return_pct"] > 0)

    # Derived features (in-memory only)
    sell["pm_abs"]    = sell["prev_momentum"].abs()
    sell["mom_delta"] = sell["momentum"] - sell["prev_momentum"]
    sell["deepening"] = (sell["momentum"] < sell["prev_momentum"])

    def _rows(grp_col: str, bins, labels) -> list:
        sell[grp_col + "_bucket"] = pd.cut(sell[grp_col], bins=bins, labels=labels)
        out = []
        for lbl in labels:
            g = sell[sell[grp_col + "_bucket"] == lbl]
            if len(g) == 0:
                continue
            out.append({
                "bucket": str(lbl),
                "n": int(len(g)),
                "win_rate": round(float(g["win"].mean()) * 100, 1),
                "avg_return": round(float(g["strategy_return_pct"].mean()), 2),
                "med_return": round(float(g["strategy_return_pct"].median()), 2),
            })
        return out

    # A. abs(prev_momentum) buckets
    abs_rows = _rows(
        "pm_abs",
        bins=[-0.001, 5, 10, 20, 30, 9999],
        labels=["0-5", "5-10", "10-20", "20-30", ">30"],
    )

    # B. mom_delta buckets
    delta_rows = _rows(
        "mom_delta",
        bins=[-9999, -10, 0, 10, 9999],
        labels=["<-10", "-10~0", "0~10", ">10"],
    )

    # C. deepening flag
    deep_rows = []
    for flag, label in [(True, "deepening"), (False, "rebounding")]:
        g = sell[sell["deepening"] == flag]
        if len(g) == 0:
            continue
        deep_rows.append({
            "bucket": label,
            "n": int(len(g)),
            "win_rate": round(float(g["win"].mean()) * 100, 1),
            "avg_return": round(float(g["strategy_return_pct"].mean()), 2),
            "med_return": round(float(g["strategy_return_pct"].median()), 2),
        })

    # D. regime × deepening cross-tab
    regime_deep_rows = []
    for regime in sell["market_regime"].dropna().unique():
        for flag, label in [(True, "deepening"), (False, "rebounding")]:
            g = sell[(sell["market_regime"] == regime) & (sell["deepening"] == flag)]
            if len(g) == 0:
                continue
            regime_deep_rows.append({
                "regime": str(regime),
                "state": label,
                "n": int(len(g)),
                "win_rate": round(float(g["win"].mean()) * 100, 1),
                "avg_return": round(float(g["strategy_return_pct"].mean()), 2),
            })

    return {
        "note": "Shadow research only — not written to CSV, no production impact",
        "n_sell": int(len(sell)),
        "pm_abs_buckets": abs_rows,
        "mom_delta_buckets": delta_rows,
        "deepening_flag": deep_rows,
        "regime_x_deepening": regime_deep_rows,
    }


def _build_rs_quantiles(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Bucket completed records by RS slope (5d) quantile and show performance."""

    slope_col = "rs_slope_5d" if "rs_slope_5d" in df.columns else "rs_slope"
    if slope_col not in df.columns or df[slope_col].isna().all():
        return []
    df = df.copy()
    df = df.dropna(subset=[slope_col])
    if len(df) < 10:
        return []

    df["rs_quantile"] = pd.qcut(df[slope_col], q=5, labels=["Q1(bottom)", "Q2", "Q3", "Q4", "Q5(top)"],
                                 duplicates="drop")
    grouped = (
        df.groupby("rs_quantile", dropna=False)
        .agg(
            sample_size=("ticker", "count"),
            win_rate=("win", "mean"),
            avg_strategy_return=("strategy_return_pct", "mean"),
            avg_rs_slope=(slope_col, "mean"),
        )
        .reset_index()
    )
    grouped["win_rate"] = grouped["win_rate"] * 100.0
    grouped = grouped.rename(columns={"rs_quantile": "bucket"})
    return grouped.to_dict("records")


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def _derive_recommendations(report: Dict[str, Any]) -> List[str]:
    recommendations: List[str] = []

    # --- Original recommendations ---
    by_type = {row["bucket"]: row for row in report.get("by_type", [])}
    buy_row = by_type.get("buy")
    sell_row = by_type.get("sell")
    if buy_row and buy_row["avg_strategy_return"] < 0:
        recommendations.append("Buy signals have negative average strategy return. Tighten entry filters or reduce exposure during weak market regimes.")
    if sell_row and sell_row["avg_strategy_return"] < 0:
        recommendations.append("Sell signals are not benefiting from downside follow-through. Recheck bearish signal definitions and short-side ranking.")

    holding_rows = report.get("by_holding_bucket", [])
    if holding_rows:
        best_bucket = max(holding_rows, key=lambda row: row["avg_strategy_return"])
        worst_bucket = min(holding_rows, key=lambda row: row["avg_strategy_return"])
        recommendations.append(
            f"Best holding window is {best_bucket['bucket']} ({best_bucket['avg_strategy_return']:.2f}%). Worst window is {worst_bucket['bucket']} ({worst_bucket['avg_strategy_return']:.2f}%). Use this to revisit exit timing."
        )

    signal_rows = report.get("by_signal", [])
    if signal_rows:
        weak_signals = [row for row in signal_rows if row["sample_size"] >= 3 and row["avg_strategy_return"] < 0]
        if weak_signals:
            names = ", ".join(f"{row['direction']}:{row['bucket']}" for row in weak_signals[:3])
            recommendations.append(f"Signals with repeat underperformance: {names}. Review the indicator thresholds behind these buckets.")

    # --- Score monotonicity check ---
    score_rows = report.get("by_score", [])
    if len(score_rows) >= 3:
        sorted_scores = sorted(score_rows, key=lambda r: r["bucket"])
        avg_returns = [r["avg_14d"] for r in sorted_scores]
        monotonic_up = all(avg_returns[i] <= avg_returns[i + 1] for i in range(len(avg_returns) - 1))
        if not monotonic_up:
            recommendations.append("Score is NOT monotonically increasing with future return. Review scoring weights — some features may be noise, not signal.")

    # --- Feature insights ---
    feat_rows = report.get("by_feature", [])
    if feat_rows:
        best = max(feat_rows, key=lambda r: r["diff_pct"])
        worst = min(feat_rows, key=lambda r: r["diff_pct"])
        if best["diff_pct"] > 2.0:
            recommendations.append(f"Best differentiating feature: {best['feature']} (True vs False: {best['diff_pct']:+.1f}% return difference). Consider increasing its weight.")
        if worst["diff_pct"] < -2.0:
            recommendations.append(f"Worst differentiating feature: {worst['feature']} (True vs False: {worst['diff_pct']:+.1f}% return difference). Consider removing or re-weighting.")

    if not recommendations:
        recommendations.append("No obvious failure cluster detected. Keep collecting history and validate by regime before changing core parameters.")
    return recommendations


def _format_section(title: str, rows: List[Dict[str, Any]], keys: List[str]) -> List[str]:
    if not rows:
        return []
    lines = ["", title]
    for row in rows:
        labels = " | ".join(f"{key}={row[key]}" for key in keys if key in row)
        lines.append(f"- {labels} | n={row['sample_size']} | win={row['win_rate']:.1f}% | avg={row['avg_strategy_return']:.2f}% | median={row['median_strategy_return']:.2f}%")
    return lines
