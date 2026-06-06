
# R100 Deriv Demo Bot — 3-Host Reliability Test Pack

This pack is for testing the same bot on Render, Railway, and Replit.

## Required project files
Upload these files together:

- `r100_live_deriv_demo_bot.py`
- `r100_ensemble_agreement_models.pkl`
- `requirements.txt`
- deployment file for the platform you are testing

## Environment variables
Set these on every host:

```bash
DERIV_DEMO_TOKEN=your Deriv virtual/demo token
LOKY_MAX_CPU_COUNT=2
```

Do not use a real token while testing. Do not add `--allow-real`.

## Common start command

```bash
python r100_live_deriv_demo_bot.py --mode high_confidence --stake 1 --max-trades 3 --warmup-candles 120 --execute-demo
```

## What to record for the reliability test
For each platform, record:

- Did the process start?
- Did it authorise the VRTC demo account?
- Did it keep streaming ticks?
- Did it sleep or restart?
- Did it reconnect after WebSocket disconnects?
- Did it log signals to `r100_live_demo_trades.csv`?
- Did it place demo contracts when `execute_decision=True`?

## Recommended test order

1. Run without `--execute-demo` first if you edit the command.
2. Then run with `--execute-demo` and `--max-trades 1`.
3. Only increase to `--max-trades 3` after one clean demo execution.

## Notes
- Render config uses a worker-style command.
- Railway config uses Dockerfile deployment.
- Replit config uses `.replit` with the direct run command.
- If a host sleeps or restarts, treat it as unreliable for this bot until proven otherwise.
