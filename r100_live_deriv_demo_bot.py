
#!/usr/bin/env python3
"""
r100_live_deriv_demo_bot.py

LIVE DEMO Deriv multiplier bot for the R_100 strategy stack.

IMPORTANT
---------
This script can place DEMO multiplier trades on Deriv if you run it with:
  --execute-demo

It is designed to refuse real accounts unless you explicitly add:
  --allow-real
Do not use --allow-real while testing.

Requirements:
  pip install websockets pandas numpy scikit-learn

Required files in same folder:
  r100_ensemble_agreement_models.pkl

Required environment variable:
  export DERIV_DEMO_TOKEN="YOUR_DEMO_API_TOKEN"

Example safe live demo run:
  python r100_live_deriv_demo_bot.py --mode high_confidence --stake 1 --max-trades 3 --execute-demo

Dry-run without placing trades:
  python r100_live_deriv_demo_bot.py --mode high_confidence --stake 1

Educational/demo research only — not financial advice.
"""

import argparse
import asyncio
import csv
import json
import os
import pickle
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import websockets


APP_ID = 1089
WS_URL = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"


class DerivWS:
    def __init__(self, token):
        self.token = token
        self.ws = None
        self.req_id = 0
        self.pending = {}

    async def connect(self):
        self.ws = await websockets.connect(WS_URL, ping_interval=20, ping_timeout=20)
        auth = await self.call({"authorize": self.token})
        return auth

    async def call(self, payload):
        self.req_id += 1
        payload = dict(payload)
        payload["req_id"] = self.req_id
        await self.ws.send(json.dumps(payload))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("req_id") == self.req_id:
                if "error" in msg:
                    raise RuntimeError(msg["error"].get("message", str(msg["error"])))
                return msg
            # Ignore subscription messages here; the main loop handles them.

    async def send(self, payload):
        self.req_id += 1
        payload = dict(payload)
        payload["req_id"] = self.req_id
        await self.ws.send(json.dumps(payload))
        return self.req_id

    async def recv(self):
        return json.loads(await self.ws.recv())


# ---------- indicators ----------

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def atr(df, n=14):
    prev = df.Close.shift(1)
    tr = pd.concat([(df.High-df.Low), (df.High-prev).abs(), (df.Low-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def bollinger(close, n=20, dev=2.0):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return mid, mid + dev * sd, mid - dev * sd


def adx(df, n=14):
    high, low, close = df.High, df.Low, df.Close
    up = high.diff(); down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([(high-low), (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr_n = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr_n
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr_n
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean(), plus_di, minus_di


def add_indicators(df):
    df = df.copy()
    df['ema20'] = ema(df.Close, 20)
    df['ema50'] = ema(df.Close, 50)
    df['ema100'] = ema(df.Close, 100)
    df['ema200'] = ema(df.Close, 200)
    df['atr14'] = atr(df, 14)
    df['bb_mid'], df['bb_upper'], df['bb_lower'] = bollinger(df.Close, 20, 2.0)
    df['adx14'], df['plus_di'], df['minus_di'] = adx(df, 14)
    df['ema_gap_atr'] = (df.ema20 - df.ema50).abs() / df.atr14
    df['regime'] = 'unknown'
    df.loc[(df.ema20 > df.ema50) & (df.ema_gap_atr >= 0.25), 'regime'] = 'bull'
    df.loc[(df.ema20 < df.ema50) & (df.ema_gap_atr >= 0.25), 'regime'] = 'bear'
    df.loc[df.ema_gap_atr < 0.25, 'regime'] = 'range'
    for L in [10, 20, 40]:
        hi = df.High.rolling(L).max().shift(1)
        lo = df.Low.rolling(L).min().shift(1)
        df[f'channel_pos_{L}'] = (df.Close - lo) / (hi - lo).replace(0, np.nan)
        df[f'donchian_mid_dist_{L}'] = (df.Close - (hi+lo)/2) / df.atr14.replace(0, np.nan)
    df['ema20_50_dist_atr'] = (df.ema20 - df.ema50) / df.atr14.replace(0, np.nan)
    df['close_ema20_dist_atr'] = (df.Close - df.ema20) / df.atr14.replace(0, np.nan)
    df['close_ema50_dist_atr'] = (df.Close - df.ema50) / df.atr14.replace(0, np.nan)
    df['bb_pos'] = (df.Close - df.bb_lower) / (df.bb_upper - df.bb_lower).replace(0, np.nan)
    df['bb_width_atr'] = (df.bb_upper - df.bb_lower) / df.atr14.replace(0, np.nan)
    df['mom_3_atr'] = (df.Close - df.Close.shift(3)) / df.atr14.replace(0, np.nan)
    df['mom_10_atr'] = (df.Close - df.Close.shift(10)) / df.atr14.replace(0, np.nan)
    return df


def candle_from_ticks(ticks):
    prices = np.array([x[1] for x in ticks], dtype=float)
    rets = np.diff(prices)
    up = int((rets > 0).sum())
    down = int((rets < 0).sum())
    abs_move = float(np.abs(rets).sum()) if len(rets) else 0.0
    net_move = float(rets.sum()) if len(rets) else 0.0
    vol = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    max_step = float(np.max(np.abs(rets))) if len(rets) else 0.0
    o, h, l, c = float(prices[0]), float(prices.max()), float(prices.min()), float(prices[-1])
    rng = max(h-l, 1e-12)
    total_dirs = up + down
    return {
        'Open': o, 'High': h, 'Low': l, 'Close': c,
        'range': h-l, 'body': c-o, 'ret': np.nan,
        'micro_up': up, 'micro_down': down,
        'micro_abs_move': abs_move,
        'micro_net_move': net_move,
        'micro_vol': vol,
        'micro_max_step': max_step,
        'micro_total_dirs': total_dirs,
        'micro_dir_balance': (up-down)/total_dirs if total_dirs else 0.0,
        'micro_efficiency': abs(net_move)/abs_move if abs_move else 0.0,
        'micro_noise_ratio': abs_move/rng if rng else 0.0,
        'close_location': (c-l)/rng if rng else 0.5,
        'upper_wick_frac': (h-max(o,c))/rng if rng else 0.0,
        'lower_wick_frac': (min(o,c)-l)/rng if rng else 0.0,
        'epoch': ticks[-1][0],
    }


def build_feature_row(df, idx, direction, strategy_id, multiplier, tp_money, sl_money, feature_cols):
    row = {}
    for c in feature_cols:
        if c in ['direction', 'multiplier', 'tp_money', 'sl_money']:
            continue
        if c in df.columns:
            row[c] = df[c].iloc[idx]
    d = float(direction)
    row['direction'] = d
    row['multiplier'] = multiplier
    row['tp_money'] = tp_money
    row['sl_money'] = sl_money
    row['dir_body'] = df['body'].iloc[idx] * d
    row['dir_micro_net'] = df['micro_net_move'].iloc[idx] * d
    row['dir_micro_balance'] = df['micro_dir_balance'].iloc[idx] * d
    row['dir_mom_3_atr'] = df['mom_3_atr'].iloc[idx] * d
    row['dir_mom_10_atr'] = df['mom_10_atr'].iloc[idx] * d
    row['trade_is_long'] = int(d == 1)
    row['strategy_id_num'] = 0 if strategy_id == 'confirmation_A' else 1
    # Fill missing expected cols
    for c in feature_cols:
        row.setdefault(c, np.nan)
    return pd.DataFrame([row])[feature_cols]


def detect_signals(df, pending):
    """Return list of candidate dicts at the latest closed candle."""
    out = []
    if len(df) < 120:  # enough for rolling micro-noise quantile and indicators
        return out, pending

    i = len(df) - 1
    L = 10
    prev_high = df.High.rolling(L).max().shift(1)
    prev_low = df.Low.rolling(L).min().shift(1)
    row = df.iloc[i]
    fail_high = row.High > prev_high.iloc[i] and row.Close < prev_high.iloc[i]
    fail_low = row.Low < prev_low.iloc[i] and row.Close > prev_low.iloc[i]

    q_noise = df.micro_noise_ratio.rolling(500, min_periods=100).quantile(0.60).iloc[i]
    high_noise = row.micro_noise_ratio >= q_noise if np.isfinite(q_noise) else False

    # First resolve pending confirmation_A from prior signal.
    new_pending = []
    for p in pending:
        if p['expires_at_index'] != i:
            continue
        confirms = (row.Close > p['signal_close']) if p['direction'] == 1 else (row.Close < p['signal_close'])
        if confirms:
            out.append({
                'strategy_id': 'confirmation_A',
                'direction': p['direction'],
                'multiplier': int(os.environ.get('BOT_MULTIPLIER', 100)),
                'tp_money': 1.0,
                'sl_money': 1.0,
                'entry_idx': i,
                'reason': 'range_adx_high_noise_not_late_next_close_breaks_signal_close',
            })
    # Pending expires after one candle.

    # Strategy B: snapback immediate range_bb + high_noise_rejection.
    range_bb_ok = row.regime == 'range' and high_noise and ((fail_high and row.High >= row.bb_upper) or (fail_low and row.Low <= row.bb_lower))
    if range_bb_ok:
        direction = -1.0 if fail_high else 1.0
        out.append({
            'strategy_id': 'snapback_B',
            'direction': direction,
            'multiplier': int(os.environ.get('BOT_MULTIPLIER', 100)),
            'tp_money': 1.5,
            'sl_money': 1.0,
            'entry_idx': i,
            'reason': 'range_bb_high_noise_immediate',
        })

    # Strategy A signal setup: range_adx + high_noise + not_late; confirm next candle.
    range_adx_ok = row.regime == 'range' and row.adx14 < 25 and high_noise and (fail_high or fail_low)
    if range_adx_ok:
        direction = -1.0 if fail_high else 1.0
        boundary = prev_high.iloc[i] if fail_high else prev_low.iloc[i]
        close_depth = (boundary - row.Close) if direction == -1 else (row.Close - boundary)
        not_late = np.isfinite(row.atr14) and row.atr14 > 0 and close_depth >= 0 and (close_depth / row.atr14 <= 0.75)
        if not_late:
            new_pending.append({
                'direction': direction,
                'signal_close': row.Close,
                'expires_at_index': i + 1,
            })

    return out, new_pending


def decision_from_models(models, feature_cols, feature_df, mode):
    probs = {}
    for name, model in models.items():
        probs[name] = float(model.predict_proba(feature_df.replace([np.inf, -np.inf], np.nan))[:, 1][0])
    avg_prob = float(np.mean(list(probs.values())))
    agreement = int(sum(p >= 0.50 for p in probs.values()))
    hybrid_prob = probs.get('hybrid_logreg', 0.0)

    if mode == 'broad':
        execute = True
    elif mode == 'balanced':
        execute = hybrid_prob >= 0.50
    elif mode == 'high_confidence':
        execute = avg_prob >= 0.60 and agreement >= 4
    else:
        raise ValueError('Live script currently supports fixed modes: high_confidence, balanced, broad')

    return execute, probs, avg_prob, agreement, hybrid_prob


async def buy_multiplier(api, symbol, direction, stake, multiplier, tp_money, sl_money, currency='USD'):
    contract_type = 'MULTUP' if direction == 1 else 'MULTDOWN'
    proposal_req = {
        'proposal': 1,
        'amount': float(stake),
        'basis': 'stake',
        'contract_type': contract_type,
        'currency': currency,
        'symbol': symbol,
        'multiplier': float(multiplier),
        'limit_order': {
            'take_profit': float(tp_money),
            'stop_loss': float(sl_money),
        },
    }
    proposal = await api.call(proposal_req)
    proposal_id = proposal['proposal']['id']
    ask_price = proposal['proposal'].get('ask_price', stake)
    buy = await api.call({'buy': proposal_id, 'price': ask_price})
    return proposal, buy


def append_csv(path, row):
    exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default='R_100')
    ap.add_argument('--mode', default='high_confidence', choices=['high_confidence', 'balanced', 'broad'])
    ap.add_argument('--stake', type=float, default=1.0)
    ap.add_argument('--currency', default='USD')
    ap.add_argument('--models', default='r100_ensemble_agreement_models.pkl')
    ap.add_argument('--execute-demo', action='store_true', help='actually place demo trades')
    ap.add_argument('--allow-real', action='store_true', help='DANGEROUS: allow non-demo account. Do not use for testing.')
    ap.add_argument('--max-trades', type=int, default=3)
    ap.add_argument('--log', default='r100_live_demo_trades.csv')
    args = ap.parse_args()

    token = os.environ.get('DERIV_DEMO_TOKEN')
    if not token:
        raise SystemExit('Set DERIV_DEMO_TOKEN first, e.g. export DERIV_DEMO_TOKEN="..."')

    with open(args.models, 'rb') as f:
        obj = pickle.load(f)
    models = obj['models']
    feature_cols = obj['feature_cols']

    api = DerivWS(token)
    auth = await api.connect()
    auth_data = auth.get('authorize', {})
    loginid = auth_data.get('loginid', '')
    balance = auth_data.get('balance')
    currency = auth_data.get('currency', args.currency)
    print(f"Authorised account: {loginid} | balance={balance} {currency}")

    if not args.allow_real and not str(loginid).upper().startswith('VRTC'):
        raise SystemExit('Refusing to trade: token does not look like a Deriv virtual/demo account (loginid should start with VRTC).')

    if not args.execute_demo:
        print('DRY RUN MODE: signals will be logged but no trades will be placed. Add --execute-demo to place demo trades.')

    await api.send({'ticks': args.symbol, 'subscribe': 1})
    tick_buffer = []
    candles = []
    pending = []
    trades_placed = 0
    open_contract_id = None

    print(f"Listening to {args.symbol} ticks. Mode={args.mode}. Max trades={args.max_trades}. Warmup=120 candles.")

    while True:
        msg = await api.recv()
        if 'error' in msg:
            print('API error:', msg['error'])
            continue
        mt = msg.get('msg_type')

        if mt == 'tick':
            tick = msg['tick']
            epoch = int(tick['epoch'])
            quote = float(tick['quote'])
            tick_buffer.append((epoch, quote))

            if len(tick_buffer) >= 30:
                candle = candle_from_ticks(tick_buffer[:30])
                tick_buffer = tick_buffer[30:]
                candles.append(candle)
                df = pd.DataFrame(candles)
                df['ret'] = df.Close.diff()
                df = add_indicators(df)

                candidates, pending = detect_signals(df, pending)
                latest = df.iloc[-1]
                print(f"{datetime.fromtimestamp(candle['epoch'], tz=timezone.utc).isoformat()} close={latest.Close:.2f} candles={len(df)} candidates={len(candidates)} pending={len(pending)}")

                for c in candidates:
                    if trades_placed >= args.max_trades:
                        print('Max trades reached. Exiting.')
                        return
                    if open_contract_id is not None:
                        print('Skipping signal because a contract is already open.')
                        continue
                    fdf = build_feature_row(df, c['entry_idx'], c['direction'], c['strategy_id'], c['multiplier'], c['tp_money'], c['sl_money'], feature_cols)
                    execute, probs, avg_prob, agreement, hybrid_prob = decision_from_models(models, feature_cols, fdf, args.mode)
                    log_row = {
                        'time_utc': datetime.now(timezone.utc).isoformat(),
                        'symbol': args.symbol,
                        'mode': args.mode,
                        'strategy_id': c['strategy_id'],
                        'direction': 'MULTUP' if c['direction'] == 1 else 'MULTDOWN',
                        'execute_decision': execute,
                        'avg_prob': avg_prob,
                        'agreement': agreement,
                        'hybrid_prob': hybrid_prob,
                        'stake': args.stake,
                        'tp_money': c['tp_money'],
                        'sl_money': c['sl_money'],
                        'reason': c['reason'],
                    }
                    for k, v in probs.items():
                        log_row[f'prob_{k}'] = v

                    print('SIGNAL:', log_row)

                    if execute and args.execute_demo:
                        try:
                            proposal, buy = await buy_multiplier(api, args.symbol, c['direction'], args.stake, c['multiplier'], c['tp_money'], c['sl_money'], currency=currency)
                            contract_id = buy['buy']['contract_id']
                            open_contract_id = contract_id
                            trades_placed += 1
                            log_row['contract_id'] = contract_id
                            log_row['buy_price'] = buy['buy'].get('buy_price')
                            log_row['status'] = 'BOUGHT'
                            print(f"BOUGHT demo contract_id={contract_id}")
                            await api.send({'proposal_open_contract': 1, 'contract_id': contract_id, 'subscribe': 1})
                        except Exception as e:
                            log_row['status'] = 'BUY_FAILED'
                            log_row['error'] = str(e)
                            print('BUY FAILED:', e)
                    else:
                        log_row['status'] = 'DRY_RUN' if not args.execute_demo else 'FILTERED_OUT'
                    append_csv(args.log, log_row)

        elif mt == 'proposal_open_contract':
            poc = msg.get('proposal_open_contract', {})
            cid = poc.get('contract_id')
            profit = poc.get('profit')
            status = poc.get('status')
            is_sold = poc.get('is_sold')
            print(f"OPEN_CONTRACT contract_id={cid} status={status} profit={profit} is_sold={is_sold}")
            if is_sold or status in ('sold', 'won', 'lost'):
                append_csv(args.log, {
                    'time_utc': datetime.now(timezone.utc).isoformat(),
                    'symbol': args.symbol,
                    'mode': args.mode,
                    'strategy_id': '',
                    'direction': '',
                    'execute_decision': '',
                    'avg_prob': '',
                    'agreement': '',
                    'hybrid_prob': '',
                    'stake': args.stake,
                    'tp_money': '',
                    'sl_money': '',
                    'reason': 'contract_closed',
                    'contract_id': cid,
                    'buy_price': '',
                    'status': status,
                    'profit': profit,
                })
                open_contract_id = None


if __name__ == '__main__':
    asyncio.run(main())
