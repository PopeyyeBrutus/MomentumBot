import requests
import pandas as pd
import numpy as np
import time
import os
import sys
from dotenv import load_dotenv

# --- LOAD ENV VARS ---
load_dotenv()

TIMEFRAME_SHORT = '1h'
TIMEFRAME_LONG = '4h'
LIMIT = 100  # candles per coin
SHORT_THRESHOLD = 3
LONG_THRESHOLD = 2.5
TOP_N = 100  # how many USDT coins to scan
DEBUG_COIN = os.getenv('DEBUG_COIN')  # Optional: for testing one coin

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Optional filters
MIN_MARKET_CAP = int(os.getenv('MIN_MARKET_CAP', 0))  # in USD
MAX_MARKET_CAP = int(os.getenv('MAX_MARKET_CAP', 10**12))
INCLUDE_CATEGORIES = os.getenv('INCLUDE_CATEGORIES', '').lower().split(',') if os.getenv('INCLUDE_CATEGORIES') else []  # e.g., ai, meme, gaming, defi, layer 1, infrastructure, stablecoins

# Cache CoinGecko ID map and metadata
COINGECKO_IDS = {}
COINGECKO_META = {}
COINGECKO_CATEGORIES_SEEN = set()  # for logging unique categories

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    response = requests.post(url, data=data)
    if response.status_code != 200:
        print(f"Telegram error {response.status_code}: {response.text}")

def fetch_klines(symbol, timeframe):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={timeframe}&limit={LIMIT}"
    resp = requests.get(url)
    data = resp.json()
    df = pd.DataFrame(data, columns=['time', 'o', 'h', 'l', 'c', 'v', '_', '_', '_', '_', '_', '_'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['c'] = df['c'].astype(float)
    df['v'] = df['v'].astype(float)
    df['h'] = df['h'].astype(float)
    df['l'] = df['l'].astype(float)
    return df[['time', 'c', 'v', 'h', 'l']]

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def macd(series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    return macd_line, signal

def bbands(series, period=20):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return lower, upper

def atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def fetch_metadata(symbol):
    base = symbol.replace("USDT", "").lower()
    if base in COINGECKO_META:
        return COINGECKO_META[base]

    try:
        if not COINGECKO_IDS:
            print("Downloading CoinGecko ID map...")
            res = requests.get("https://api.coingecko.com/api/v3/coins/list")
            for item in res.json():
                COINGECKO_IDS[item['symbol'].lower()] = item['id']

        coingecko_id = COINGECKO_IDS.get(base)
        if not coingecko_id:
            return {}

        url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}"
        res = requests.get(url)
        meta = res.json()
        out = {
            'market_cap': meta['market_data']['market_cap'].get('usd', 0),
            'categories': meta.get('categories', [])
        }
        COINGECKO_META[base] = out
        return out

    except Exception as e:
        print(f"Metadata error for {symbol}: {e}")
        return {}

def analyze_coin(symbol, timeframe, rsi_floor=30, volume_mult=2):
    try:
        df = fetch_klines(symbol, timeframe)
        df.set_index('time', inplace=True)
        if len(df) < 2:
            raise ValueError("Not enough data to evaluate indicators")

        df['EMA_21'] = ema(df['c'], 21)
        df['EMA_50'] = ema(df['c'], 50)
        df['RSI'] = rsi(df['c'])
        df['MACD'], df['MACD_signal'] = macd(df['c'])
        df['BBL'], df['BBU'] = bbands(df['c'])
        df['ATR'] = atr(df['h'], df['l'], df['c'])

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        score = 0
        reasons = []

        if prev['RSI'] < rsi_floor and latest['RSI'] > prev['RSI']:
            score += 1
            reasons.append(f"RSI: {latest['RSI']:.1f} ↗")

        if prev['MACD'] < prev['MACD_signal'] and latest['MACD'] > latest['MACD_signal']:
            score += 1.5
            reasons.append("MACD crossover")

        if latest['c'] > latest['EMA_21'] > latest['EMA_50']:
            score += 1
            reasons.append("EMA stack")

        if latest['c'] > latest['BBL'] and latest['c'] > latest['BBU']:
            score += 1
            reasons.append("BB breakout")

        avg_volume = df['v'][-20:].mean()
        if latest['v'] > volume_mult * avg_volume:
            score += 2
            reasons.append(f"Volume spike: {latest['v']:.2f}")

        if latest['ATR'] > prev['ATR']:
            score += 1
            reasons.append("ATR rising")

        return score, reasons, latest['c']

    except Exception as e:
        print(f"Error with {symbol} on {timeframe}: {e}")
        return 0, [], 0

def get_usdt_pairs():
    # Fetch Binance pairs
    url = "https://api.binance.com/api/v3/exchangeInfo"
    data = requests.get(url).json()
    usdt_pairs = [s['symbol'] for s in data['symbols'] if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING']

    # Build CoinGecko ID map first
    global COINGECKO_IDS
    if not COINGECKO_IDS:
        print("Downloading CoinGecko ID map...")
        res = requests.get("https://api.coingecko.com/api/v3/coins/list")
        for item in res.json():
            COINGECKO_IDS[item['symbol'].lower()] = item['id']

    # Get top marketcap coins from CoinGecko
    print("Fetching top marketcap coins from CoinGecko...")
    res = requests.get("https://api.coingecko.com/api/v3/coins/markets", params={
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 400,
        'page': 1
    })
    cg_data = res.json()

    valid = []
    for coin in cg_data:
        symbol = coin['symbol'].upper() + 'USDT'
        if symbol in usdt_pairs:
            COINGECKO_META[coin['symbol']] = {
                'market_cap': coin.get('market_cap', 0),
                'categories': []  # optional: update later
            }
            valid.append(symbol)
        if len(valid) >= TOP_N:
            break

    return valid

pairs = [DEBUG_COIN] if DEBUG_COIN else get_usdt_pairs()
for symbol in pairs:
    score_1h, reasons_1h, price_1h = analyze_coin(symbol, TIMEFRAME_SHORT, rsi_floor=30, volume_mult=2)
    score_4h, reasons_4h, price_4h = analyze_coin(symbol, TIMEFRAME_LONG, rsi_floor=40, volume_mult=1.3)

    print(f"Checked {symbol}: 1h Score = {score_1h}, 4h Score = {score_4h}")

    if score_1h >= SHORT_THRESHOLD and score_4h >= LONG_THRESHOLD:
        meta = fetch_metadata(symbol)
        mcap_val = meta.get('market_cap', 0)
        cats_raw = meta.get('categories', [])
        COINGECKO_CATEGORIES_SEEN.update([c.lower() for c in cats_raw])

        if not (MIN_MARKET_CAP <= mcap_val <= MAX_MARKET_CAP):
            print(f"{symbol} skipped: market cap {mcap_val} not in range")
            continue

        if INCLUDE_CATEGORIES:
            cat_match = any(cat.lower() in [c.lower() for c in cats_raw] for cat in INCLUDE_CATEGORIES)
            if not cat_match:
                print(f"{symbol} skipped: no matching category")
                continue

        mcap = f"${mcap_val:,.0f}"
        cats = ", ".join(cats_raw[:3])

        msg = (
            f"🚀 High-Confluence Signal: {symbol}\n"
            f"Price: ${price_1h:.4f}\n"
            f"Market Cap: {mcap}\n"
            f"Category: {cats}\n\n"
            f"🕐 1h Score: {score_1h}\n" + '\n'.join([f"✅ {r}" for r in reasons_1h]) + "\n\n"
            f"🕓 4h Score: {score_4h}\n" + '\n'.join([f"📈 {r}" for r in reasons_4h])
        )
        print(msg)
        send_telegram_alert(msg)

print("\n👁️ Unique categories seen this run:")
print(', '.join(sorted(COINGECKO_CATEGORIES_SEEN)))
time.sleep(1.2)

