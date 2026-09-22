Squeeze Tracking Analysis
- Total records: 75
- Completed records: 50
- Active records: 25
- Date range: 2026-04-21 to 2026-09-22

By Type
- bucket=sell | n=26 | win=65.4% | avg=1.18% | median=2.21%
- bucket=buy | n=24 | win=37.5% | avg=-4.52% | median=-8.08%

By Signal
- direction=buy | bucket=強烈買入 (爆發) | n=9 | win=33.3% | avg=3.17% | median=-7.99%
- direction=sell | bucket=強烈賣出 (跌破) | n=8 | win=62.5% | avg=2.00% | median=1.34%
- direction=sell | bucket=賣出 (動能轉弱) | n=18 | win=66.7% | avg=0.82% | median=3.67%
- direction=buy | bucket=買入 (動能增強) | n=15 | win=40.0% | avg=-9.13% | median=-8.18%

By Holding Day
- bucket=82 | n=5 | win=60.0% | avg=24.52% | median=2.17%
- bucket=28 | n=23 | win=60.9% | avg=0.74% | median=2.24%
- bucket=30 | n=2 | win=50.0% | avg=0.43% | median=0.43%
- bucket=81 | n=20 | win=40.0% | avg=-10.91% | median=-5.18%

By Regime
- bucket=bull_trend | direction=sell | n=26 | win=65.4% | avg=1.18% | median=2.21%
- bucket=bull_trend | direction=buy | n=24 | win=37.5% | avg=-4.52% | median=-8.08%

By Holding Bucket
- bucket=11-14d | direction=sell | n=26 | win=65.4% | avg=1.18% | median=2.21%
- bucket=11-14d | direction=buy | n=24 | win=37.5% | avg=-4.52% | median=-8.08%

Pattern Combination Performance
  Combo                                  n  Avg 14D   Win%
  Squeeze+Houyi+Whale                   50    -1.6%  52.0%

[Shadow Research] SELL prev_momentum Analysis (n=26, research only — not in production)
  Shadow research only — not written to CSV, no production impact

  A. abs(prev_momentum) × SELL win rate
  abs(pm)      n   win%     avg     med
  0-5         14   57.1   -2.23    1.83
  5-10         1    0.0  -13.36  -13.36
  10-20        3   66.7    3.02    8.40
  20-30        2  100.0    3.67    3.67
  >30          6   83.3    9.82   14.71

  B. mom_delta (momentum − prev_momentum) × SELL win rate
  delta        n   win%     avg     med
  <-10         6   66.7    5.89    9.87
  -10~0       19   63.2   -0.32    2.24
  0~10         1  100.0    1.49    1.49

  C. Deepening flag (momentum < prev_momentum)
  state          n   win%     avg     med
  deepening     24   66.7    5.30    3.67
  rebounding     2   50.0  -48.23  -48.23

  D. Regime × Deepening × SELL
  regime          state          n   win%     avg
  bull_trend      deepening     24   66.7    5.30
  bull_trend      rebounding     2   50.0  -48.23

Recommendations
- Buy signals have negative average strategy return. Tighten entry filters or reduce exposure during weak market regimes.
- Best holding window is 11-14d (1.18%). Worst window is 11-14d (-4.52%). Use this to revisit exit timing.
- Signals with repeat underperformance: buy:買入 (動能增強). Review the indicator thresholds behind these buckets.