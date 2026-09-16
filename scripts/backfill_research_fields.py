#!/usr/bin/env python3
"""
backfill_research_fields.py
============================
Research-only backfill for three missing field groups in recommendations.csv.

Fields filled:
  - market_regime   : ^TWII MA20/MA60 state as-of the signal DATE (historical, no look-ahead)
  - return_5d/10d/14d/20d : forward returns from entry_price on signal date
  - has_squeeze / has_houyi / has_whale : re-run pattern detection on signal date bar

Constraints (HARD):
  - Production ranking (ranking_score, experimental_score) UNCHANGED
  - feature_schema_version stays "phase6a"
  - No new filter conditions written

Usage:
  # Small-batch dry-run (first 5 completed rows, print only):
  python scripts/backfill_research_fields.py --dry-run --limit 5

  # Full backfill (writes to CSV, keeps backup):
  python scripts/backfill_research_fields.py

  # Only specific field groups:
  python scripts/backfill_research_fields.py --fields regime returns patterns

Author: research pipeline (non-production)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

# ── Path setup ───────────────────────────────────────────────────────────────
# Force squeeze-tw-screener/src to the front of sys.path.
# Also evict any squeeze-cn-screener paths that may have been installed into
# site-packages or sys.path by a sibling editable install — they shadow the
# tw-screener's src and expose an older patterns.py without benchmark_close.
ROOT = Path(__file__).resolve().parents[1]      # .../squeeze-tw-screener
SRC  = ROOT / "src"                              # .../squeeze-tw-screener/src

# Remove any path that contains "squeeze-cn-screener" to avoid shadowing
sys.path = [p for p in sys.path if "squeeze-cn-screener" not in p]
# Prepend tw-screener src so it wins over any installed egg/wheel
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
else:
    sys.path.remove(str(SRC))
    sys.path.insert(0, str(SRC))

# Verify we're loading from the right place before importing squeeze modules
_check_path = SRC / "squeeze" / "engine" / "patterns.py"
if not _check_path.exists():
    raise RuntimeError(f"Expected patterns.py at {_check_path} — wrong ROOT?")

from squeeze.data.downloader import download_market_data
from squeeze.engine.indicators import calculate_squeeze_indicators
from squeeze.engine.patterns import detect_squeeze, detect_houyi_shooting_sun, detect_whale_trading

# ── Sanity-check: patterns must accept benchmark_close ───────────────────────
import inspect as _inspect
_sq_params = list(_inspect.signature(detect_squeeze).parameters.keys())
if "benchmark_close" not in _sq_params:
    raise RuntimeError(
        f"detect_squeeze loaded from wrong module (params={_sq_params}). "
        f"Expected src from {SRC}. Check sys.path ordering."
    )
del _inspect, _sq_params, _check_path


# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
BENCHMARK = "^TWII"
HISTORY_PERIOD = "2y"        # enough to cover all historical signals
TRADING_DAYS_FORWARD = 25    # fetch this many forward bars beyond signal date
# Fields we write — must NOT include ranking or schema version fields
BACKFILL_REGIME_COL   = "market_regime"
BACKFILL_RETURN_COLS  = ["return_5d", "return_10d", "return_14d", "return_20d"]
BACKFILL_PATTERN_COLS = ["has_squeeze", "has_houyi", "has_whale"]

# Regime classification mirrors _infer_market_context() but uses historical date
def _regime_on_date(bm_close: pd.Series, as_of: pd.Timestamp) -> str:
    """
    Compute ^TWII regime as of `as_of` date.
    Uses MA20 / MA60 of closes UP TO AND INCLUDING as_of (no look-ahead).
    Returns: 'bull_trend' | 'bear_trend' | 'range_bound' | 'unknown'
    """
    # Slice: only dates <= as_of
    hist = bm_close[bm_close.index.normalize() <= as_of.normalize()]
    if len(hist) < 20:
        return "unknown"
    close_now  = float(hist.iloc[-1])
    ma20 = float(hist.rolling(20).mean().iloc[-1])
    ma60 = float(hist.rolling(60).mean().iloc[-1]) if len(hist) >= 60 else ma20
    # 20-day return (look-back only)
    ret_20 = (close_now / float(hist.iloc[-21]) - 1.0) if len(hist) >= 21 else 0.0
    if close_now > ma20 > ma60 and ret_20 > 0:
        return "bull_trend"
    if close_now < ma20 < ma60 and ret_20 < 0:
        return "bear_trend"
    return "range_bound"


def _forward_returns(ticker_hist: pd.DataFrame,
                     entry_date: pd.Timestamp,
                     entry_price: float) -> dict[str, Any]:
    """
    Compute forward returns from entry_date+1 trading day open/close.
    Uses CLOSE prices only; no same-day data is used (entry_price is from scan).

    Anti look-ahead:
      - Finds the first bar AFTER entry_date in the history.
      - Windows (5/10/14/20) are counted as TRADING DAYS from that bar.
    """
    result: dict[str, Any] = {c: np.nan for c in BACKFILL_RETURN_COLS}
    if ticker_hist.empty or entry_price <= 0:
        return result

    idx = ticker_hist.index.normalize()
    # First bar strictly after entry_date
    future_mask = idx > entry_date.normalize()
    if not future_mask.any():
        return result

    future = ticker_hist[future_mask].copy()
    windows = {5: "return_5d", 10: "return_10d", 14: "return_14d", 20: "return_20d"}
    for n, col in windows.items():
        # iloc[n-1] = price n trading-days after entry (bar 0 = first day after entry)
        if len(future) >= n:
            px = float(future["Close"].iloc[n - 1])
            result[col] = round((px / entry_price - 1.0) * 100, 4)
    return result


def _pattern_on_date(ticker: str,
                     ticker_hist: pd.DataFrame,
                     bm_close: pd.Series,
                     as_of: pd.Timestamp) -> dict[str, Any]:
    """
    Re-run pattern detection on the bar at `as_of` date.
    Uses ONLY data up to and including as_of (no look-ahead).

    Returns dict with has_squeeze, has_houyi, has_whale.
    Note: benchmark_close passed positionally to avoid stale __pycache__ mismatch.
    """
    result = {"has_squeeze": np.nan, "has_houyi": np.nan, "has_whale": np.nan}
    if ticker_hist.empty:
        return result

    # Slice history up to as_of (inclusive) — no look-ahead
    hist_slice = ticker_hist[ticker_hist.index.normalize() <= as_of.normalize()].copy()
    if len(hist_slice) < 30:
        log.debug(f"  {ticker}: insufficient history ({len(hist_slice)} bars) at {as_of.date()}")
        return result

    # Slice benchmark up to as_of — no look-ahead
    bm_slice = bm_close[bm_close.index.normalize() <= as_of.normalize()]
    bm_arg = bm_slice if len(bm_slice) > 20 else None

    # Use positional args throughout to avoid stale .pyc keyword-mismatch errors
    try:
        sq = detect_squeeze(hist_slice, bm_arg)
        result["has_squeeze"] = bool(sq.get("is_squeezed", False))
    except Exception as exc:
        log.warning(f"  detect_squeeze failed for {ticker} @ {as_of.date()}: {exc}")

    try:
        hy = detect_houyi_shooting_sun(hist_slice, benchmark_close=bm_slice if len(bm_slice) > 20 else None)
        result["has_houyi"] = bool(hy.get("is_houyi", False))
    except Exception as exc:
        log.warning(f"  detect_houyi failed for {ticker} @ {as_of.date()}: {exc}")

    try:
        # detect_whale_trading signature: (df) or (df, benchmark_close)?
        import inspect
        sig = inspect.signature(detect_whale_trading)
        if "benchmark_close" in sig.parameters:
            wh = detect_whale_trading(hist_slice, benchmark_close=bm_slice if len(bm_slice) > 20 else None)
        else:
            wh = detect_whale_trading(hist_slice)
        result["has_whale"] = bool(wh.get("is_whale", False))
    except Exception as exc:
        log.warning(f"  detect_whale failed for {ticker} @ {as_of.date()}: {exc}")

    return result


# ── Validation helpers ───────────────────────────────────────────────────────

def validate_backfill(df_before: pd.DataFrame, df_after: pd.DataFrame) -> list[str]:
    """
    Run validation checks on before/after DataFrames.
    Returns list of PASS/FAIL strings for each check.
    """
    reports: list[str] = []

    # 1. market_regime has > 1 unique value
    regimes = df_after["market_regime"].dropna().unique()
    if len(regimes) > 1:
        reports.append(f"PASS  market_regime: {len(regimes)} distinct values {sorted(regimes)}")
    else:
        reports.append(f"FAIL  market_regime: still only {regimes} — check benchmark data")

    # 2. return_Nd NaN rate improved for completed rows
    completed = df_after[df_after["status"] == "completed"]
    for col in BACKFILL_RETURN_COLS:
        before_nan = df_before.loc[df_before["status"] == "completed", col].isna().mean()
        after_nan  = completed[col].isna().mean()
        tag = "PASS" if after_nan < before_nan else "WARN"
        reports.append(
            f"{tag}  {col}: NaN {before_nan*100:.0f}% → {after_nan*100:.0f}%"
        )

    # 3. Pattern booleans: NaN reduced
    for col in BACKFILL_PATTERN_COLS:
        before_nan = df_before.loc[df_before["status"] == "completed", col].isna().mean()
        after_nan  = completed[col].isna().mean()
        tag = "PASS" if after_nan < before_nan else "WARN"
        reports.append(
            f"{tag}  {col}: NaN {before_nan*100:.0f}% → {after_nan*100:.0f}%"
        )

    # 4. Production ranking columns UNCHANGED
    rank_cols = ["ranking_score", "experimental_score", "feature_schema_version"]
    for col in rank_cols:
        if col not in df_before.columns or col not in df_after.columns:
            continue
        changed = (df_before[col].astype(str) != df_after[col].astype(str)).sum()
        tag = "PASS" if changed == 0 else "FAIL"
        reports.append(f"{tag}  {col}: {changed} rows changed (must be 0)")

    # 5. No look-ahead: return_5d should be NaN for rows with < 5 trading days in future
    # (approximate check: days_tracked < 5 → return_5d should be NaN or already filled)
    early = df_after[(df_after["status"] == "completed") & (df_after["days_tracked"] < 5)]
    if len(early) > 0:
        n_filled = early["return_5d"].notna().sum()
        tag = "WARN" if n_filled > 0 else "PASS"
        reports.append(f"{tag}  look-ahead guard: {n_filled} rows with days_tracked<5 have return_5d filled")

    return reports


# ── Main backfill logic ──────────────────────────────────────────────────────

def backfill(
    csv_path: Path,
    fields: list[str],
    dry_run: bool = False,
    limit: int | None = None,
) -> None:
    """
    Main backfill routine.

    Args:
        csv_path : path to recommendations.csv
        fields   : subset of ['regime', 'returns', 'patterns']
        dry_run  : if True, print diffs but do not write
        limit    : process only the first N completed rows (for smoke test)
    """
    do_regime   = "regime"   in fields
    do_returns  = "returns"  in fields
    do_patterns = "patterns" in fields

    log.info(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    df_backup = df.copy()   # keep for validation

    # Only backfill completed rows (tracking rows will fill naturally)
    target_mask = df["status"] == "completed"
    if limit is not None:
        # Take first `limit` completed rows
        target_idx = df[target_mask].head(limit).index
        target_mask = df.index.isin(target_idx)

    n_target = target_mask.sum()
    log.info(f"Target rows: {n_target} completed")

    # ── Fetch benchmark history once ─────────────────────────────────────────
    log.info(f"Fetching benchmark {BENCHMARK} ({HISTORY_PERIOD}) …")
    try:
        bm_raw = yf.download(BENCHMARK, period=HISTORY_PERIOD, interval="1d", progress=False)
        bm_close: pd.Series = bm_raw["Close"].squeeze()
        bm_close.index = pd.to_datetime(bm_close.index).tz_localize(None)
        log.info(f"Benchmark: {len(bm_close)} bars, {bm_close.index[0].date()} – {bm_close.index[-1].date()}")
    except Exception as exc:
        log.error(f"Cannot fetch benchmark: {exc}")
        sys.exit(1)

    # ── Collect unique tickers and signal dates ──────────────────────────────
    target_rows = df[target_mask].copy()
    unique_tickers = target_rows["ticker"].unique().tolist()
    log.info(f"Fetching price history for {len(unique_tickers)} tickers …")

    # Download history for all tickers at once
    ticker_histories: dict[str, pd.DataFrame] = {}
    try:
        raw = download_market_data(unique_tickers, period=HISTORY_PERIOD)
        if raw.empty:
            raise ValueError("Empty response from download_market_data")
        # Normalise index timezone
        if raw.index.tz is not None:
            raw.index = raw.index.tz_localize(None)
        for t in unique_tickers:
            try:
                if len(unique_tickers) == 1:
                    th = raw.dropna(subset=["Close"]).copy()
                else:
                    if t not in raw.columns.get_level_values(0):
                        continue
                    th = raw[t].dropna(subset=["Close"]).copy()
                th.index = pd.to_datetime(th.index).tz_localize(None)
                ticker_histories[t] = th
            except Exception as exc:
                log.warning(f"  Cannot extract {t}: {exc}")
    except Exception as exc:
        log.error(f"Batch download failed: {exc}")
        sys.exit(1)

    log.info(f"Successfully loaded {len(ticker_histories)}/{len(unique_tickers)} tickers")

    # ── Row-by-row backfill ──────────────────────────────────────────────────
    n_regime_filled = 0
    n_return_filled = 0
    n_pattern_filled = 0

    for idx in target_rows.index:
        row = df.loc[idx]
        ticker = row["ticker"]
        date_str = str(row["date"])

        try:
            as_of = pd.Timestamp(date_str)
        except Exception:
            log.warning(f"  Row {idx} ({ticker}): invalid date '{date_str}', skipping")
            continue

        entry_price = float(row.get("entry_price", 0) or 0)
        th = ticker_histories.get(ticker)

        # ── market_regime ────────────────────────────────────────────────────
        if do_regime:
            regime = _regime_on_date(bm_close, as_of)
            if not dry_run:
                df.at[idx, BACKFILL_REGIME_COL] = regime
            else:
                log.info(f"  [dry] {ticker} {date_str}: market_regime = {regime}")
            n_regime_filled += 1

        # ── forward returns ──────────────────────────────────────────────────
        if do_returns and th is not None:
            # Only fill if currently NaN (don't overwrite already-computed values)
            any_nan = any(pd.isna(row.get(c)) for c in BACKFILL_RETURN_COLS)
            if any_nan:
                returns = _forward_returns(th, as_of, entry_price)
                if not dry_run:
                    for col, val in returns.items():
                        if pd.isna(row.get(col)) and not np.isnan(val):
                            df.at[idx, col] = val
                else:
                    log.info(f"  [dry] {ticker} {date_str}: returns = {returns}")
                n_return_filled += 1

        # ── pattern booleans ─────────────────────────────────────────────────
        if do_patterns and th is not None:
            any_nan = any(pd.isna(row.get(c)) for c in BACKFILL_PATTERN_COLS)
            if any_nan:
                patterns = _pattern_on_date(ticker, th, bm_close, as_of)
                if not dry_run:
                    for col, val in patterns.items():
                        if pd.isna(row.get(col)) and not np.isnan(float(val) if isinstance(val, bool) else (val if val is not None else float("nan"))):
                            df.at[idx, col] = val
                else:
                    log.info(f"  [dry] {ticker} {date_str}: patterns = {patterns}")
                n_pattern_filled += 1

    log.info(f"Processed: regime={n_regime_filled}, returns={n_return_filled}, patterns={n_pattern_filled}")

    # ── Dry-run: print stats, do not write ───────────────────────────────────
    if dry_run:
        log.info("Dry-run mode: no files written.")
        return

    # ── Backup original CSV ──────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = csv_path.with_suffix(f".backup_{ts}.csv")
    shutil.copy2(csv_path, backup_path)
    log.info(f"Backup written: {backup_path.name}")

    # ── Write updated CSV ─────────────────────────────────────────────────────
    df.to_csv(csv_path, index=False)
    log.info(f"Updated CSV written: {csv_path.name}")

    # ── Validation report ────────────────────────────────────────────────────
    df_after = pd.read_csv(csv_path)
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)

    checks = validate_backfill(df_backup, df_after)
    for c in checks:
        print(f"  {c}")

    # ── Provenance metadata ───────────────────────────────────────────────────
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_sha = "unknown"

    print("\n" + "-"*60)
    print("PROVENANCE")
    print(f"  timestamp          : {ts}")
    print(f"  git_sha            : {git_sha}")
    print(f"  csv_path           : {csv_path}")
    print(f"  backup             : {backup_path.name}")
    print(f"  fields_backfilled  : {fields}")
    print(f"  feature_schema_ver : phase6a (unchanged)")
    print(f"  ranking_version    : v1 (unchanged)")

    # ── Missing-rate summary ─────────────────────────────────────────────────
    print("\n" + "-"*60)
    print("MISSING RATE SUMMARY (completed rows)")
    done_after = df_after[df_after["status"] == "completed"]
    done_before = df_backup[df_backup["status"] == "completed"]
    cols_to_report = [BACKFILL_REGIME_COL] + BACKFILL_RETURN_COLS + BACKFILL_PATTERN_COLS
    for col in cols_to_report:
        before_nan = done_before[col].isna().mean() * 100 if col in done_before else float("nan")
        after_nan  = done_after[col].isna().mean() * 100 if col in done_after else float("nan")
        print(f"  {col:<30} {before_nan:5.1f}% → {after_nan:5.1f}% NaN")

    # ── Distribution summary ─────────────────────────────────────────────────
    print("\n" + "-"*60)
    print("DISTRIBUTION (after, completed rows)")
    if BACKFILL_REGIME_COL in df_after.columns:
        print(f"\n  market_regime:\n{done_after['market_regime'].value_counts(dropna=False).to_string()}")
    for col in BACKFILL_RETURN_COLS:
        if col in df_after.columns:
            s = done_after[col].dropna()
            if len(s):
                print(f"\n  {col}: n={len(s)}, mean={s.mean():.2f}%, median={s.median():.2f}%, "
                      f"min={s.min():.2f}%, max={s.max():.2f}%")
    for col in BACKFILL_PATTERN_COLS:
        if col in df_after.columns:
            vc = done_after[col].value_counts(dropna=False)
            print(f"\n  {col}: {vc.to_dict()}")

    print("="*60)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill research fields in recommendations.csv")
    parser.add_argument(
        "--csv", default="recommendations.csv",
        help="Path to tracking CSV (default: recommendations.csv)"
    )
    parser.add_argument(
        "--fields", nargs="+",
        choices=["regime", "returns", "patterns"],
        default=["regime", "returns", "patterns"],
        help="Which field groups to backfill (default: all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would change without writing files"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only first N completed rows (smoke test)"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    backfill(
        csv_path=csv_path,
        fields=args.fields,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
