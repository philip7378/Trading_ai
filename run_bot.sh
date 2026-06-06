#!/usr/bin/env bash
set -e
export LOKY_MAX_CPU_COUNT=${LOKY_MAX_CPU_COUNT:-2}
python r100_live_deriv_demo_bot.py --mode ${BOT_MODE:-high_confidence} --stake ${BOT_STAKE:-1} --max-trades ${BOT_MAX_TRADES:-3} --warmup-candles ${BOT_WARMUP_CANDLES:-120} --execute-demo
