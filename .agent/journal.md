# Full-send journal: the forecast step

Goal (Gabriel, 2026-11-08): first the small fixes in one PR (issue #2), then the whole forecasting step built, tested, reviewed and merged; report back when ready.
Reading: away · done · feature · logic. Base: main at bba1532.

## Lanes
- PR A, branch fix/weather-narrow-redownload, worktree D:/code/.worktrees/iberian-battery-15min-fix-weather-narrow-redownload: issue #2 only. Skipped from "small items": the newest-run weather refinement (it changes the data, not small) and the archive/cleanup (end of the paper).
- PR B, branch feat/forecast, this worktree: `ib15 forecast`. Design in docs/direction.md (2026-11-08 entries).

## Decisions
- Sample frozen at the 404 days built (2025-10-01..2026-11-08).
- Forecast: naive / expert ARX / expert+weather, per local quarter-hour OLS, rolling 56-day window, hourly = mean of 4 qh forecasts, main spec only, settled at actual prices, DM test for weather.
- Workers via the Agent tool (foreground, concurrent), worktrees made by the orchestrator under D:/code/.worktrees per CLAUDE.md.

## Rounds
(none yet)

## Open
