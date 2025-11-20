"""
Intraday Live Trading Terminal - Enhanced Version

This file includes:
- Full Nifty 50 & 100 scanning
- Limited auto-execution (10 confirmed trades max)
- Enhanced signal display with entry time and current price
- Improved trade management
"""

import time
from datetime import datetime, time as dt_time

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# Configuration
st.set_page_config(page_title="Intraday Terminal Pro - Enhanced", layout="wide")
IND_TZ = pytz.timezone("Asia/Kolkata")

CAPITAL = 1_000_000.0
TRADE_ALLOC = 0.15
MAX_DAILY_TRADES = 10
MAX_STOCK_TRADES = 10
MAX_AUTO_TRADES = 10  # Maximum auto-execution trades

SIGNAL_REFRESH_MS = 60000
PRICE_REFRESH_MS = 3000

MARKET_OPTIONS = ["CASH"]

# Enhanced Nifty 50 & 100 Lists
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS",
    "SBIN.NS", "ASIANPAINT.NS", "HCLTECH.NS", "AXISBANK.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NTPC.NS",
    "NESTLEIND.NS", "POWERGRID.NS", "M&M.NS", "BAJFINANCE.NS", "ONGC.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIPORTS.NS", "COALINDIA.NS",
    "HDFCLIFE.NS", "DRREDDY.NS", "HINDALCO.NS", "CIPLA.NS", "SBILIFE.NS",
    "GRASIM.NS", "TECHM.NS", "BAJAJFINSV.NS", "BRITANNIA.NS", "EICHERMOT.NS",
    "DIVISLAB.NS", "SHREECEM.NS", "APOLLOHOSP.NS", "UPL.NS", "BAJAJ-AUTO.NS",
    "HEROMOTOCO.NS", "INDUSINDBK.NS", "ADANIENT.NS", "TATACONSUM.NS", "BPCL.NS"
]

# Extended Nifty 100 (including Nifty 50 + additional stocks)
NIFTY_100 = NIFTY_50 + [
    "HDFC.NS", "BAJAJHLDNG.NS", "TATAMOTORS.NS", "VEDANTA.NS", "PIDILITIND.NS",
    "BERGEPAINT.NS", "AMBUJACEM.NS", "DABUR.NS", "HAVELLS.NS", "ICICIPRULI.NS",
    "MARICO.NS", "PEL.NS", "SIEMENS.NS", "TORNTPHARM.NS", "ACC.NS",
    "AUROPHARMA.NS", "BOSCHLTD.NS", "GLENMARK.NS", "MOTHERSUMI.NS", "BIOCON.NS",
    "CADILAHC.NS", "COLPAL.NS", "CONCOR.NS", "DLF.NS", "GODREJCP.NS",
    "HINDPETRO.NS", "IBULHSGFIN.NS", "IOC.NS", "JINDALSTEL.NS", "LUPIN.NS",
    "MANAPPURAM.NS", "MCDOWELL-N.NS", "NMDC.NS", "PETRONET.NS", "PFC.NS",
    "PNB.NS", "RBLBANK.NS", "SAIL.NS", "SRTRANSFIN.NS", "TATAPOWER.NS",
    "YESBANK.NS", "ZEEL.NS"
]

# Utilities

def now_indian():
    return datetime.now(IND_TZ)

def market_open():
    n = now_indian()
    try:
        open_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(9, 15)))
        close_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 30)))
        return open_time <= n <= close_time
    except Exception:
        return False

def should_auto_close():
    n = now_indian()
    try:
        auto_close_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 10)))
        return n >= auto_close_time
    except Exception:
        return False

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rs = rs.fillna(0)
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k.fillna(50), d.fillna(50)

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(close, period=20, std_dev=2):
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_market_profile_vectorized(high, low, close, volume, bins=20):
    low_val = min(low.min(), close.min())
    high_val = max(high.max(), close.max())
    if np.isclose(low_val, high_val):
        high_val = low_val * 1.01 if low_val != 0 else 1.0
    edges = np.linspace(low_val, high_val, bins + 1)
    hist, _ = np.histogram(close, bins=edges, weights=volume)
    centers = (edges[:-1] + edges[1:]) / 2
    if hist.sum() == 0:
        poc = float(close.iloc[-1])
        va_high = poc * 1.01
        va_low = poc * 0.99
        profile = [{"price": float(c), "volume": int(v)} for c, v in zip(centers, hist)]
        return {"poc": float(poc), "value_area_high": float(va_high), "value_area_low": float(va_low), "profile": profile}
    idx = int(np.argmax(hist))
    poc = float(centers[idx])
    sorted_idx = np.argsort(hist)[::-1]
    cumulative = 0.0
    total = float(hist.sum())
    selected = []
    for i in sorted_idx:
        selected.append(centers[i])
        cumulative += hist[i]
        if cumulative / total >= 0.70:
            break
    va_high = float(max(selected))
    va_low = float(min(selected))
    profile = [{"price": float(c), "volume": int(v)} for c, v in zip(centers, hist)]
    return {"poc": poc, "value_area_high": va_high, "value_area_low": va_low, "profile": profile}

def calculate_support_resistance_advanced(high, low, close, period=20):
    resistance = []
    support = []
    ln = len(high)
    if ln < period * 2 + 1:
        return {"support": float(close.iloc[-1] * 0.98), "resistance": float(close.iloc[-1] * 1.02),
                "support_levels": [], "resistance_levels": []}
    for i in range(period, ln - period):
        if high.iloc[i] >= high.iloc[i - period:i + period + 1].max():
            resistance.append(float(high.iloc[i]))
        if low.iloc[i] <= low.iloc[i - period:i + period + 1].min():
            support.append(float(low.iloc[i]))
    recent_res = sorted(resistance)[-3:] if resistance else [float(close.iloc[-1] * 1.02)]
    recent_sup = sorted(support)[:3] if support else [float(close.iloc[-1] * 0.98)]
    return {"support": float(np.mean(recent_sup)), "resistance": float(np.mean(recent_res)),
            "support_levels": recent_sup, "resistance_levels": recent_res}

# Data Manager

class EnhancedDataManager:
    def __init__(self):
        self.price_cache = {}
        self.signal_cache = {}

    def _validate_live_price(self, symbol):
        now_ts = time.time()
        key = f"price_{symbol}"
        if key in self.price_cache:
            cached = self.price_cache[key]
            if now_ts - cached["ts"] < 2:
                return cached["price"]
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="2d", interval="1m")
            if df is not None and not df.empty:
                price = float(df["Close"].iloc[-1])
                self.price_cache[key] = {"price": round(price, 2), "ts": now_ts}
                return round(price, 2)
            df = ticker.history(period="2d", interval="5m")
            if df is not None and not df.empty:
                price = float(df["Close"].iloc[-1])
                self.price_cache[key] = {"price": round(price, 2), "ts": now_ts}
                return round(price, 2)
        except Exception:
            pass
        known = {"RELIANCE.NS": 2750.0, "TCS.NS": 3850.0, "HDFCBANK.NS": 1650.0}
        base = known.get(symbol, 1000.0)
        self.price_cache[key] = {"price": float(base), "ts": now_ts}
        return float(base)

    @st.cache_data(ttl=30)
    def _fetch_yf(_self, symbol, period, interval):
        try:
            return yf.download(symbol, period=period, interval=interval, progress=False)
        except Exception:
            return pd.DataFrame()

    def get_stock_data(self, symbol, interval="15m"):
        if interval == "1m":
            period = "1d"
        elif interval == "5m":
            period = "2d"
        elif interval == "15m":
            period = "7d"
        else:
            period = "14d"

        df = self._fetch_yf(symbol, period, interval)
        if df is None or df.empty or len(df) < 20:
            return self.create_validated_demo_data(symbol)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        expected = ["Open", "High", "Low", "Close", "Volume"]
        for e in expected:
            if e not in df.columns:
                if e.upper() in df.columns:
                    df[e] = df[e.upper()]
                else:
                    return self.create_validated_demo_data(symbol)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
        if len(df) < 20:
            return self.create_validated_demo_data(symbol)

        try:
            live = self._validate_live_price(symbol)
            df.iloc[-1, df.columns.get_loc("Close")] = live
            df.iloc[-1, df.columns.get_loc("High")] = max(df.iloc[-1]["High"], live)
            df.iloc[-1, df.columns.get_loc("Low")] = min(df.iloc[-1]["Low"], live)
        except Exception:
            pass

        df["EMA8"] = ema(df["Close"], 8)
        df["EMA21"] = ema(df["Close"], 21)
        df["EMA50"] = ema(df["Close"], 50)
        df["RSI14"] = rsi(df["Close"], 14).fillna(50)
        df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"]).fillna(0)
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(df["Close"])
        df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bollinger_bands(df["Close"])
        df["Stoch_K"], df["Stoch_D"] = stochastic(df["High"], df["Low"], df["Close"])
        df["VWAP"] = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()

        mp = calculate_market_profile_vectorized(df["High"], df["Low"], df["Close"], df["Volume"], bins=24)
        df["POC"] = mp["poc"]
        df["VA_High"] = mp["value_area_high"]
        df["VA_Low"] = mp["value_area_low"]

        sr = calculate_support_resistance_advanced(df["High"], df["Low"], df["Close"])
        df["Support"] = sr["support"]
        df["Resistance"] = sr["resistance"]

        return df

    def create_validated_demo_data(self, symbol):
        live = self._validate_live_price(symbol)
        periods = 200
        end = now_indian()
        dates = pd.date_range(end=end, periods=periods, freq="15min")
        base = float(live)
        rng = np.random.default_rng(int(abs(hash(symbol)) % (2 ** 32 - 1)))
        returns = rng.normal(0, 0.0008, periods)
        prices = base * np.cumprod(1 + returns)
        openp = prices * (1 + rng.normal(0, 0.001, periods))
        highp = prices * (1 + abs(rng.normal(0, 0.004, periods)))
        lowp = prices * (1 - abs(rng.normal(0, 0.004, periods)))
        vol = rng.integers(20000, 500000, periods)
        df = pd.DataFrame({"Open": openp, "High": highp, "Low": lowp, "Close": prices, "Volume": vol}, index=dates)
        df.iloc[-1, df.columns.get_loc("Close")] = live
        df["EMA8"] = ema(df["Close"], 8)
        df["EMA21"] = ema(df["Close"], 21)
        df["EMA50"] = ema(df["Close"], 50)
        df["RSI14"] = rsi(df["Close"], 14).fillna(50)
        df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"]).fillna(0)
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(df["Close"])
        df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bollinger_bands(df["Close"])
        df["Stoch_K"], df["Stoch_D"] = stochastic(df["High"], df["Low"], df["Close"])
        df["VWAP"] = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()
        mp = calculate_market_profile_vectorized(df["High"], df["Low"], df["Close"], df["Volume"], bins=24)
        df["POC"] = mp["poc"]
        df["VA_High"] = mp["value_area_high"]
        df["VA_Low"] = mp["value_area_low"]
        sr = calculate_support_resistance_advanced(df["High"], df["Low"], df["Close"])
        df["Support"] = sr["support"]
        df["Resistance"] = sr["resistance"]
        return df

# Trading Engine

class EnhancedIntradayTrader:
    def __init__(self, capital=CAPITAL):
        self.initial_capital = float(capital)
        self.cash = float(capital)
        self.positions = {}
        self.trade_log = []
        self.daily_trades = 0
        self.stock_trades = 0
        self.auto_trades_count = 0  # Track auto-executed trades
        self.last_reset = now_indian().date()
        self.selected_market = "CASH"
        self.auto_execution = False
        self.signal_history = []
        self.auto_close_triggered = False

    def reset_daily_counts(self):
        current_date = now_indian().date()
        if current_date != self.last_reset:
            self.daily_trades = 0
            self.stock_trades = 0
            self.auto_trades_count = 0
            self.last_reset = current_date

    def can_auto_trade(self):
        """Check if auto trading is allowed within limits"""
        return (self.auto_trades_count < MAX_AUTO_TRADES and 
                self.daily_trades < MAX_DAILY_TRADES and
                market_open())

    def calculate_support_resistance(self, symbol, current_price):
        try:
            data = data_manager.get_stock_data(symbol, "15m")
            if data is None or len(data) < 20:
                return current_price * 0.98, current_price * 1.02
            return float(data["Support"].iloc[-1]), float(data["Resistance"].iloc[-1])
        except Exception:
            return current_price * 0.98, current_price * 1.02

    def calculate_intraday_target_sl(self, entry_price, action, atr, current_price, support, resistance):
        if atr <= 0 or np.isnan(atr):
            atr = max(entry_price * 0.005, 1.0)
        if action == "BUY":
            target = min(entry_price * 1.018, resistance * 0.998)
            stop_loss = max(entry_price * 0.988, support * 1.002)
            target = max(target, entry_price + (atr * 1.5))
            stop_loss = min(stop_loss, entry_price - (atr * 1.0))
        else:
            target = max(entry_price * 0.982, support * 1.002)
            stop_loss = min(entry_price * 1.012, resistance * 0.998)
            target = min(target, entry_price - (atr * 1.5))
            stop_loss = max(stop_loss, entry_price + (atr * 1.0))
        rr = abs(target - entry_price) / max(abs(entry_price - stop_loss), 1e-6)
        if rr < 0.8:
            if action == "BUY":
                target = entry_price + max((entry_price - stop_loss) * 1.2, (atr * 1.5))
            else:
                target = entry_price - max((stop_loss - entry_price) * 1.2, (atr * 1.5))
        return round(float(target), 2), round(float(stop_loss), 2)

    def equity(self):
        total = float(self.cash)
        for symbol, pos in self.positions.items():
            if pos.get("status") == "OPEN":
                try:
                    data = data_manager.get_stock_data(symbol, "5m")
                    price = float(data["Close"].iloc[-1]) if data is not None and len(data) > 0 else pos["entry_price"]
                    total += pos["quantity"] * price
                except Exception:
                    total += pos["quantity"] * pos["entry_price"]
        return total

    def execute_trade(self, symbol, action, quantity, price, stop_loss=None, target=None, win_probability=0.75, auto_trade=False):
        self.reset_daily_counts()
        
        if self.daily_trades >= MAX_DAILY_TRADES:
            return False, "Daily trade limit reached"
        if self.stock_trades >= MAX_STOCK_TRADES:
            return False, "Stock trade limit reached"
        if auto_trade and self.auto_trades_count >= MAX_AUTO_TRADES:
            return False, "Auto trade limit reached"
            
        trade_value = float(quantity) * float(price)
        if action == "BUY" and trade_value > self.cash:
            return False, "Insufficient capital"
            
        trade_id = f"TRADE_{symbol}_{len(self.trade_log)}_{int(time.time())}"
        record = {
            "trade_id": trade_id, 
            "symbol": symbol, 
            "action": action, 
            "quantity": int(quantity),
            "entry_price": float(price), 
            "stop_loss": float(stop_loss) if stop_loss else None,
            "target": float(target) if target else None, 
            "timestamp": now_indian(),
            "status": "OPEN", 
            "current_pnl": 0.0, 
            "current_price": float(price),
            "win_probability": float(win_probability), 
            "closed_pnl": 0.0,
            "entry_time": now_indian().strftime("%H:%M:%S"),
            "auto_trade": auto_trade
        }
        
        if action == "BUY":
            self.positions[symbol] = record
            self.cash -= trade_value
        else:
            margin = trade_value * 0.2
            record["margin_used"] = margin
            self.positions[symbol] = record
            self.cash -= margin
            
        self.stock_trades += 1
        self.trade_log.append(record)
        self.daily_trades += 1
        
        if auto_trade:
            self.auto_trades_count += 1
            
        return True, f"{'[AUTO] ' if auto_trade else ''}{action} {int(quantity)} {symbol} @ ₹{price:.2f}"

    def update_positions_pnl(self):
        if should_auto_close() and not self.auto_close_triggered:
            self.auto_close_all_positions()
            self.auto_close_triggered = True
            return
        for symbol, pos in list(self.positions.items()):
            if pos.get("status") != "OPEN":
                continue
            try:
                data = data_manager.get_stock_data(symbol, "5m")
                if data is not None and len(data) > 0:
                    price = float(data["Close"].iloc[-1])
                    pos["current_price"] = price
                    entry = pos["entry_price"]
                    if pos["action"] == "BUY":
                        pnl = (price - entry) * pos["quantity"]
                    else:
                        pnl = (entry - price) * pos["quantity"]
                    pos["current_pnl"] = float(pnl)
                    pos["max_pnl"] = max(pos.get("max_pnl", 0.0), float(pnl))
                    sl = pos.get("stop_loss")
                    tg = pos.get("target")
                    if sl is not None:
                        if (pos["action"] == "BUY" and price <= sl) or (pos["action"] == "SELL" and price >= sl):
                            self.close_position(symbol, exit_price=sl)
                            continue
                    if tg is not None:
                        if (pos["action"] == "BUY" and price >= tg) or (pos["action"] == "SELL" and price <= tg):
                            self.close_position(symbol, exit_price=tg)
                            continue
            except Exception:
                continue

    def auto_close_all_positions(self):
        for sym in list(self.positions.keys()):
            self.close_position(sym)

    def close_position(self, symbol, exit_price=None):
        if symbol not in self.positions:
            return False, "Position not found"
        pos = self.positions[symbol]
        if exit_price is None:
            try:
                data = data_manager.get_stock_data(symbol, "5m")
                exit_price = float(data["Close"].iloc[-1]) if data is not None and len(data) > 0 else pos["entry_price"]
            except Exception:
                exit_price = pos["entry_price"]
        if pos["action"] == "BUY":
            pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
            self.cash += pos["quantity"] * exit_price
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["quantity"]
            self.cash += pos.get("margin_used", 0) + (pos["quantity"] * pos["entry_price"])
        pos["status"] = "CLOSED"
        pos["exit_price"] = float(exit_price)
        pos["closed_pnl"] = float(pnl)
        pos["exit_time"] = now_indian()
        try:
            del self.positions[symbol]
        except Exception:
            pass
        return True, f"Closed {symbol} @ ₹{exit_price:.2f} | P&L: ₹{pnl:+.2f}"

    def get_open_positions_data(self):
        self.update_positions_pnl()
        out = []
        for symbol, pos in self.positions.items():
            if pos.get("status") != "OPEN":
                continue
            try:
                data = data_manager.get_stock_data(symbol, "5m")
                price = float(data["Close"].iloc[-1]) if data is not None and len(data) > 0 else pos["entry_price"]
                if pos["action"] == "BUY":
                    pnl = (price - pos["entry_price"]) * pos["quantity"]
                else:
                    pnl = (pos["entry_price"] - price) * pos["quantity"]
                var = ((price - pos["entry_price"]) / pos["entry_price"]) * 100
                sup, res = self.calculate_support_resistance(symbol, price)
                out.append({
                    "Symbol": symbol.replace(".NS", ""), 
                    "Action": pos["action"], 
                    "Quantity": pos["quantity"],
                    "Entry Price": f"₹{pos['entry_price']:.2f}", 
                    "Current Price": f"₹{price:.2f}",
                    "P&L": f"₹{pnl:+.2f}", 
                    "Variance %": f"{var:+.2f}%", 
                    "Stop Loss": f"₹{pos.get('stop_loss', 0):.2f}",
                    "Target": f"₹{pos.get('target', 0):.2f}", 
                    "Support": f"₹{sup:.2f}", 
                    "Resistance": f"₹{res:.2f}",
                    "Win %": f"{pos.get('win_probability', 0.75)*100:.1f}%", 
                    "Entry Time": pos.get("entry_time"),
                    "Auto Trade": "Yes" if pos.get("auto_trade") else "No",
                    "Status": pos.get("status")
                })
            except Exception:
                continue
        return out

    def get_performance_stats(self):
        self.update_positions_pnl()
        closed = [t for t in self.trade_log if t.get("status") == "CLOSED"]
        total_trades = len(closed)
        open_pnl = sum([p.get("current_pnl", 0) for p in self.positions.values() if p.get("status") == "OPEN"])
        if total_trades == 0:
            return {
                "total_trades": 0, 
                "win_rate": 0.0, 
                "total_pnl": 0.0, 
                "avg_pnl": 0.0, 
                "open_positions": len(self.positions),
                "open_pnl": open_pnl,
                "auto_trades": self.auto_trades_count
            }
        wins = len([t for t in closed if t.get("closed_pnl", 0) > 0])
        total_pnl = sum([t.get("closed_pnl", 0) for t in closed])
        win_rate = wins / total_trades if total_trades else 0.0
        avg_pnl = total_pnl / total_trades if total_trades else 0.0
        
        auto_trades = [t for t in self.trade_log if t.get("auto_trade")]
        auto_closed = [t for t in auto_trades if t.get("status") == "CLOSED"]
        auto_win_rate = len([t for t in auto_closed if t.get("closed_pnl", 0) > 0]) / len(auto_closed) if auto_closed else 0.0
        
        return {
            "total_trades": total_trades, 
            "win_rate": win_rate, 
            "total_pnl": total_pnl, 
            "avg_pnl": avg_pnl,
            "open_positions": len(self.positions), 
            "open_pnl": open_pnl,
            "auto_trades": self.auto_trades_count,
            "auto_win_rate": auto_win_rate
        }

    def generate_quality_signals(self, universe, max_scan=None, min_confidence=0.8, min_score=7):
        signals = []
        stocks = NIFTY_50 if universe == "Nifty 50" else NIFTY_100
        
        # Scan all stocks in the universe
        if max_scan is None:
            max_scan = len(stocks)
            
        weights = {"ema_trend": 3, "vwap": 2, "poc": 2, "macd": 2, "rsi": 1, "stoch": 1, "volume": 2}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, symbol in enumerate(stocks[:max_scan]):
            try:
                status_text.text(f"Scanning {symbol} ({idx+1}/{len(stocks[:max_scan])})")
                progress_bar.progress((idx + 1) / len(stocks[:max_scan]))
                
                data = data_manager.get_stock_data(symbol, "15m")
                if data is None or len(data) < 30:
                    continue
                    
                live = float(data["Close"].iloc[-1])
                ema8 = float(data["EMA8"].iloc[-1])
                ema21 = float(data["EMA21"].iloc[-1])
                ema50 = float(data["EMA50"].iloc[-1])
                rsi_val = float(data["RSI14"].iloc[-1])
                atr = float(data["ATR"].iloc[-1])
                macd_line = float(data["MACD"].iloc[-1])
                macd_signal = float(data["MACD_Signal"].iloc[-1])
                stoch_k = float(data["Stoch_K"].iloc[-1])
                stoch_d = float(data["Stoch_D"].iloc[-1])
                vwap = float(data["VWAP"].iloc[-1])
                poc = float(data["POC"].iloc[-1])
                va_high = float(data["VA_High"].iloc[-1])
                va_low = float(data["VA_Low"].iloc[-1])
                support = float(data["Support"].iloc[-1])
                resistance = float(data["Resistance"].iloc[-1])
                vol_latest = float(data["Volume"].iloc[-1])
                vol_avg = float(data["Volume"].rolling(20).mean().iloc[-1]) if len(data["Volume"]) >= 20 else float(data["Volume"].mean())
                volume_spike = vol_latest > vol_avg * 1.3
                
                # Bullish scoring
                score = 0
                ema_trend = ema8 > ema21 > ema50
                if ema_trend:
                    score += weights["ema_trend"]
                vwap_cond = (live > vwap) and (va_low <= live <= va_high)
                if vwap_cond:
                    score += weights["vwap"]
                poc_cond = live > poc
                if poc_cond:
                    score += weights["poc"]
                macd_cond = (macd_line > macd_signal) and (macd_line > 0)
                if macd_cond:
                    score += weights["macd"]
                rsi_cond = 40 <= rsi_val <= 65
                if rsi_cond:
                    score += weights["rsi"]
                stoch_cond = (stoch_k > stoch_d) and (20 <= stoch_k <= 80)
                if stoch_cond:
                    score += weights["stoch"]
                if volume_spike:
                    score += weights["volume"]
                    
                max_possible = sum(weights.values())
                confidence = score / max_possible
                action = None
                confirmed = False
                
                if score >= min_score and ema_trend and macd_cond:
                    action = "BUY"
                    confirmed = True
                    
                # Bearish scoring
                bearish_score = 0
                if ema8 < ema21 < ema50:
                    bearish_score += weights["ema_trend"]
                if live < vwap and va_low <= live <= va_high:
                    bearish_score += weights["vwap"]
                if live < poc:
                    bearish_score += weights["poc"]
                if macd_line < macd_signal and macd_line < 0:
                    bearish_score += weights["macd"]
                if 35 <= rsi_val <= 60:
                    bearish_score += weights["rsi"]
                if stoch_k < stoch_d and 20 <= stoch_k <= 80:
                    bearish_score += weights["stoch"]
                if vol_latest > vol_avg * 1.3:
                    bearish_score += weights["volume"]
                    
                if bearish_score >= min_score and bearish_score > score:
                    action = "SELL"
                    score = bearish_score
                    confidence = bearish_score / max_possible
                    confirmed = True
                    
                if confirmed and confidence >= min_confidence:
                    entry = live
                    target, stop_loss = self.calculate_intraday_target_sl(entry, action, atr, live, support, resistance)
                    rr = abs(target - entry) / max(abs(entry - stop_loss), 1e-6)
                    if rr < 1.0:
                        continue
                    win_prob = min(0.90, 0.70 + (confidence - min_confidence) * 0.5)
                    potential_pnl = abs(target - entry) * 250.0
                    
                    signals.append({
                        "symbol": symbol, 
                        "action": action, 
                        "entry": float(entry),
                        "current_price": float(live), 
                        "target": float(target), 
                        "stop_loss": float(stop_loss),
                        "confidence": float(confidence), 
                        "win_probability": float(win_prob),
                        "rsi": float(rsi_val), 
                        "potential_pnl": float(potential_pnl),
                        "risk_reward": float(rr), 
                        "atr": float(atr), 
                        "support": float(support),
                        "resistance": float(resistance), 
                        "score": int(score), 
                        "vwap": float(vwap),
                        "poc": float(poc), 
                        "confirmed": True,
                        "entry_time": now_indian().strftime("%H:%M:%S"),
                        "volume_ratio": vol_latest / vol_avg if vol_avg > 0 else 1.0
                    })
                    
            except Exception as e:
                continue
                
        progress_bar.empty()
        status_text.empty()
        
        signals.sort(key=lambda x: (x["score"], x["confidence"]), reverse=True)
        self.signal_history = signals[:15]  # Keep more signals in history
        
        return signals[:10]  # Return top 10 signals

    def auto_execute_signals(self, signals):
        """Auto-execute top signals within limits"""
        executed = []
        for signal in signals[:3]:  # Auto-execute top 3 signals only
            if not self.can_auto_trade():
                break
                
            if signal["symbol"] in self.positions:
                continue  # Skip if already in position
                
            qty = int((self.cash * TRADE_ALLOC) / signal["entry"])
            if qty > 0:
                success, msg = self.execute_trade(
                    symbol=signal["symbol"], 
                    action=signal["action"], 
                    quantity=qty, 
                    price=signal["entry"], 
                    stop_loss=signal["stop_loss"], 
                    target=signal["target"], 
                    win_probability=signal["win_probability"],
                    auto_trade=True
                )
                if success:
                    executed.append(msg)
                    
        return executed

# Initialize components
data_manager = EnhancedDataManager()

if "trader" not in st.session_state:
    st.session_state.trader = EnhancedIntradayTrader()
trader = st.session_state.trader

# Streamlit UI

st.markdown("<h1 style='text-align:center;'>Intraday Terminal Pro - Enhanced</h1>", unsafe_allow_html=True)
st_autorefresh(interval=PRICE_REFRESH_MS, key="price_refresh_enhanced")

# Top metrics
cols = st.columns(7)
try:
    nift = data_manager._validate_live_price("^NSEI")
    cols[0].metric("NIFTY 50", f"₹{nift:,.2f}")
except Exception:
    cols[0].metric("NIFTY 50", "N/A")
try:
    bn = data_manager._validate_live_price("^NSEBANK")
    cols[1].metric("BANK NIFTY", f"₹{bn:,.2f}")
except Exception:
    cols[1].metric("BANK NIFTY", "N/A")
cols[2].metric("Market Status", "LIVE" if market_open() else "CLOSED")
cols[3].metric("Auto Close", "15:10")
cols[4].metric("Stock Trades", f"{trader.stock_trades}/{MAX_STOCK_TRADES}")
cols[5].metric("Auto Trades", f"{trader.auto_trades_count}/{MAX_AUTO_TRADES}")
cols[6].metric("Available Cash", f"₹{trader.cash:,.0f}")

# Sidebar configuration
st.sidebar.header("Trading Configuration")
trader.selected_market = st.sidebar.selectbox("Market Type", MARKET_OPTIONS)
trader.auto_execution = st.sidebar.checkbox("Auto Execution", value=False)
min_conf_percent = st.sidebar.slider("Minimum Confidence %", 70, 95, 80, 5)
min_score = st.sidebar.slider("Minimum Score", 6, 12, 7, 1)
scan_limit = st.sidebar.selectbox("Scan Limit", ["All Stocks", "Top 40", "Top 20"], index=0)

# Convert scan limit to number
max_scan_map = {"All Stocks": None, "Top 40": 40, "Top 20": 20}
max_scan = max_scan_map[scan_limit]

# Main tabs
tabs = st.tabs(["Dashboard", "Signals", "Paper Trading", "History", "Charts"])

with tabs[0]:
    st.subheader("Account Summary")
    trader.update_positions_pnl()
    perf = trader.get_performance_stats()
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Account Value", f"₹{trader.equity():,.0f}", delta=f"₹{trader.equity() - trader.initial_capital:+,.0f}")
    c2.metric("Available Cash", f"₹{trader.cash:,.0f}")
    c3.metric("Open Positions", len(trader.positions))
    c4.metric("Open P&L", f"₹{perf['open_pnl']:+.2f}")
    c5.metric("Win Rate", f"{perf['win_rate']:.1%}")
    
    if perf['auto_trades'] > 0:
        st.metric("Auto Trade Win Rate", f"{perf.get('auto_win_rate', 0):.1%}")

with tabs[1]:
    st.subheader("Quality Signals")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        universe = st.selectbox("Universe", ["Nifty 50", "Nifty 100"])
        generate_btn = st.button("Generate Signals", type="primary")
    
    with col2:
        if trader.auto_execution:
            st.info("🔴 Auto Execution: ACTIVE (Max 10 trades)")
        else:
            st.info("⚪ Auto Execution: INACTIVE")
    
    if generate_btn or trader.auto_execution:
        with st.spinner(f"Scanning {universe} stocks for signals..."):
            signals = trader.generate_quality_signals(
                universe, 
                max_scan=max_scan,
                min_confidence=min_conf_percent / 100.0, 
                min_score=min_score
            )
        
        if signals:
            # Enhanced signal display with more information
            signal_data = []
            for s in signals:
                signal_data.append({
                    "Symbol": s["symbol"].replace(".NS", ""),
                    "Action": s["action"],
                    "Entry Price": f"₹{s['entry']:.2f}",
                    "Current Price": f"₹{s['current_price']:.2f}",
                    "Target": f"₹{s['target']:.2f}", 
                    "Stop Loss": f"₹{s['stop_loss']:.2f}",
                    "Confidence": f"{s['confidence']:.1%}",
                    "Win %": f"{s['win_probability']:.1%}",
                    "R:R": f"{s['risk_reward']:.2f}",
                    "Score": s["score"],
                    "RSI": f"{s['rsi']:.1f}",
                    "Entry Time": s["entry_time"],
                    "Volume Ratio": f"{s['volume_ratio']:.2f}x"
                })
            
            df_signals = pd.DataFrame(signal_data)
            st.dataframe(df_signals, use_container_width=True)
            
            # Auto-execution
            if trader.auto_execution and trader.can_auto_trade():
                executed = trader.auto_execute_signals(signals)
                if executed:
                    st.success("Auto-execution completed:")
                    for msg in executed:
                        st.write(f"✓ {msg}")
                    st.rerun()
            
            # Manual execution section
            st.subheader("Manual Execution")
            for s in signals:
                col_a, col_b, col_c = st.columns([3, 1, 1])
                with col_a:
                    st.write(f"**{s['symbol'].replace('.NS','')}** - {s['action']} @ ₹{s['entry']:.2f} | "
                           f"Target: ₹{s['target']:.2f} | Stop: ₹{s['stop_loss']:.2f} | "
                           f"R:R: {s['risk_reward']:.2f} | Score: {s['score']}")
                
                with col_b:
                    qty = int((trader.cash * TRADE_ALLOC) / s["entry"])
                    st.write(f"Qty: {qty}")
                
                with col_c:
                    if st.button(f"Execute", key=f"exec_{s['symbol']}"):
                        success, msg = trader.execute_trade(
                            symbol=s["symbol"], 
                            action=s["action"], 
                            quantity=qty, 
                            price=s["entry"], 
                            stop_loss=s["stop_loss"], 
                            target=s["target"], 
                            win_probability=s["win_probability"]
                        )
                        if success:
                            st.success(msg)
                            st.rerun()
        else:
            st.info("No confirmed signals found with current criteria.")

with tabs[2]:
    st.subheader("Paper Trading")
    trader.update_positions_pnl()
    open_pos = trader.get_open_positions_data()
    
    if open_pos:
        st.dataframe(pd.DataFrame(open_pos), use_container_width=True)
        
        st.write("Close positions:")
        cols_close = st.columns(4)
        for idx, symbol in enumerate(list(trader.positions.keys())):
            with cols_close[idx % 4]:
                if st.button(f"Close {symbol}", key=f"close_{symbol}"):
                    success, msg = trader.close_position(symbol)
                    if success:
                        st.success(msg)
                        st.rerun()
        
        if st.button("Close All Positions", type="primary"):
            for sym in list(trader.positions.keys()):
                trader.close_position(sym)
            st.rerun()
    else:
        st.info("No open positions.")

with tabs[3]:
    st.subheader("Trade History")
    if trader.trade_log:
        hist = []
        for t in trader.trade_log:
            hist.append({
                "Symbol": t["symbol"].replace(".NS", ""), 
                "Action": t["action"], 
                "Qty": t["quantity"],
                "Entry": f"₹{t['entry_price']:.2f}", 
                "Exit": f"₹{t.get('exit_price','N/A')}", 
                "P&L": f"₹{t.get('closed_pnl', t.get('current_pnl', 0)):+.2f}", 
                "Status": t["status"],
                "Auto": "Yes" if t.get("auto_trade") else "No",
                "Entry Time": t.get("entry_time", "N/A")
            })
        st.dataframe(pd.DataFrame(hist), use_container_width=True)
        
        # Performance summary
        perf = trader.get_performance_stats()
        st.metric("Overall Win Rate", f"{perf['win_rate']:.1%}")
        
    else:
        st.info("No trades executed yet.")

with tabs[4]:
    st.subheader("Charts")
    left, right = st.columns([1, 3])
    with left:
        symbol = st.selectbox("Select Stock", NIFTY_100)  # Include all Nifty 100 stocks
        interval = st.selectbox("Interval", ["5m", "15m", "30m"])
    with right:
        chart_data = data_manager.get_stock_data(symbol, interval)
        if chart_data is not None and len(chart_data) > 10:
            cp = chart_data["Close"].iloc[-1]
            st.write(f"{symbol.replace('.NS','')} - {interval} | Live Price: ₹{cp:.2f}")
            
            fig = make_subplots(
                rows=3, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.08, 
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=("Price & Market Profile", "MACD", "RSI & Volume")
            )
            
            # Price chart
            fig.add_trace(go.Candlestick(
                x=chart_data.index, 
                open=chart_data["Open"], 
                high=chart_data["High"], 
                low=chart_data["Low"], 
                close=chart_data["Close"], 
                name="Price"
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["POC"], name="POC", line=dict(width=1, dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["VA_High"], name="VA High", line=dict(width=1, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["VA_Low"], name="VA Low", line=dict(width=1, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["Support"], name="Support", line=dict(width=1, dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["Resistance"], name="Resistance", line=dict(width=1, dash="dash")), row=1, col=1)
            
            # MACD
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["MACD"], name="MACD"), row=2, col=1)
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["MACD_Signal"], name="Signal"), row=2, col=1)
            fig.add_trace(go.Bar(x=chart_data.index, y=chart_data["MACD_Hist"], name="MACD Hist"), row=2, col=1)
            
            # RSI and Volume
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["RSI14"], name="RSI"), row=3, col=1)
            fig.add_trace(go.Bar(x=chart_data.index, y=chart_data["Volume"], name="Volume"), row=3, col=1)
            
            fig.update_layout(xaxis_rangeslider_visible=False, height=700)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Loading chart data...")

st.markdown("---")
st.markdown("<div style='text-align:center;'>Enhanced Intraday Terminal - Full Nifty Scan & Auto Execution</div>", unsafe_allow_html=True)
