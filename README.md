# makefun-gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Hardi24/makefun-gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Hardi24/makefun-gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast GTA VI's launch-day Twitch viewers hour by hour — a statistical quantile model built by an AI agent for make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty, $250 in USDC.

`main.py` reads one case file (`<id>.in`: hour-1 print plus 38–42 past game launches as comparables, with SullyGnome hourly Twitch data) and writes `<id>.out` with five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) for each hour 2–24 of release day.

How it works: per-hour weighted least squares of log viewers on log hour-1 viewers and log pre-launch peak, over clean comparables only (excluding simultaneous-global unlocks), platform-weighted toward console-only launches like GTA VI (PS5 + Xbox Series). Slopes are shrunk toward their structural targets and the intercept recentered. Hours 5–9 — the US-unlock jump region — are fitted on unlock-aligned time: the US-East midnight unlock lands at hour 5 in daylight time (04:00 UTC) and hour 6 in standard time (05:00 UTC); GTA VI releases 19 Nov 2026, after DST ends, so the jump is at hour 6. Quantiles come from the regression's empirical residuals, narrowed 0.35x (calibrated on practice). A saturation cap freezes the size effect at the largest comparable, so an hour-1 print above every past launch forecasts like the biggest launches instead of extrapolating the size trend downward.

Practice validation (4 past launches: Wolverine, Red Dead Redemption 2, God of War Ragnarök, Spider-Man 2): mean pinball loss on the log scale **0.1374** (per case: 0.122 / 0.136 / 0.107 / 0.184), vs the house baseline's 0.3871. Deterministic, standard library only, no network, ~0.1 s per case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=b8hax0j7&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
