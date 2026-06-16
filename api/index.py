Let's go. Here's the first file:

---

## `api/index.py`

```python
import os
import sys
import time
import base64

# Make project root importable so strategy, symbols, etc. resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, send_from_directory

from symbols import resolve_symbol_exact, DISPLAY_NAMES
from strategy import (
    compute_indicators, generate_signal, confidence_score,
    risk_reward, daily_pct_change, fetch_daily_data, daily_trend,
    invalidation_conditions, fetch_timeframe, TIMEFRAMES, DEFAULT_TIMEFRAME,
    support_resistance, confidence_breakdown, market_regime, setup_grade,
    passes_verification, timeframe_lean, CONFIRM_TIMEFRAMES,
)
from pnl import log_trade, today_total, daily_totals
from news import upcoming_high_impact_usd_events
from chart import render_candlestick_png
from data_provider import fetch_with_active_provider, load_config as load_provider_config, \
    save_config as save_provider_config, PROVIDERS
from watchlist import load_watchlist, add_symbol, remove_symbol, set_pinned
import tracker

PUBLIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public")

app = Flask(__name__, static_folder=None)

_daily_cache = {}
_news_cache = (None, None)
CACHE_TTL = 900
_last_bias_by_key = {}


def get_daily_trend(yf_symbol):
    cached = _daily_cache.get(yf_symbol)
    if cached and time.time() - cached[1] < CACHE_TTL:
        return cached[0]
    try:
        trend = daily_trend(fetch_daily_data(yf_symbol))
    except Exception:
        trend = None
    _daily_cache[yf_symbol] = (trend, time.time())
    return trend


def get_news_events():
    global _news_cache
    events, fetched_at = _news_cache
    if fetched_at and time.time() - fetched_at < CACHE_TTL:
        return events
    try:
        events = upcoming_high_impact_usd_events()
    except Exception:
        events = []
    _news_cache = (events, time.time())
    return events


def action_label(bias, last_bias):
    if bias == "LONG":
        return "GO LONG"
    if bias == "SHORT":
        return "GO SHORT"
    if last_bias == "LONG":
        return "SELL"
    return "WAIT"


_price_cache = {}
PRICE_CACHE_TTL = 4


def get_price_data(yf_symbol, timeframe):
    cache_key = (yf_symbol, timeframe)
    cached = _price_cache.get(cache_key)
    if cached and time.time() - cached[1] < PRICE_CACHE_TTL:
        return cached[0]
    if timeframe == DEFAULT_TIMEFRAME:
        df = fetch_with_active_provider(yf_symbol)
    else:
        df = fetch_timeframe(yf_symbol, timeframe)
    _price_cache[cache_key] = (df, time.time())
    return df


def evaluate_due_signals(alias, timeframe, current_price):
    for row in tracker.due_for_evaluation():
        if row["symbol"] == alias and row["timeframe"] == timeframe:
            tracker.evaluate_signal(row["id"], current_price)


def multi_timeframe_confirmation(yf_symbol, timeframe, bias):
    confirm_list = CONFIRM_TIMEFRAMES.get(timeframe, [])
    if not confirm_list or bias == "NO TRADE / WAIT":
        return None, 0, 0, []
    agree = 0
    detail = []
    for tf in confirm_list:
        raw_df = get_price_data(yf_symbol, tf)
        lean = timeframe_lean(raw_df)
        if lean is None:
            continue
        if lean == bias:
            agree += 1
            detail.append(f"{tf} agrees ({lean})")
        else:
            detail.append(f"{tf} disagrees ({lean})")
    total = len(detail)
    if total == 0:
        return None, 0, 0, []
    return round(agree / total * 100), agree, total, detail


def build_signal(raw_symbol, timeframe=DEFAULT_TIMEFRAME, include_chart=True, dark_mode=True):
    alias, yf_symbol = resolve_symbol_exact(raw_symbol)
    if not yf_symbol:
        return {"error": f"Unrecognized symbol '{raw_symbol}'"}
    if timeframe not in TIMEFRAMES:
        timeframe = DEFAULT_TIMEFRAME

    df = get_price_data(yf_symbol, timeframe)
    if df is None or len(df) < 25:
        return {"error": f"Not enough data for {alias} yet"}

    df = compute_indicators(df)
    trend = get_daily_trend(yf_symbol)
    raw_bias, score, reasons, last = generate_signal(df, daily_trend_direction=trend)
    confidence, sample_size, winrate = confidence_score(df, raw_bias, score, daily_trend_direction=trend)
    regime = market_regime(last)
    if regime:
        reasons = reasons + [f"market regime: {regime}"]

    current_price = float(last["Close"])
    evaluate_due_signals(alias, timeframe, current_price)

    real_adjustment = tracker.symbol_timeframe_confidence_adjustment(alias, timeframe)
    if real_adjustment and raw_bias != "NO TRADE / WAIT":
        confidence = max(0, min(100, confidence + real_adjustment))
        reasons = reasons + [f"real tracked {timeframe} history for {alias}: "
                              f"{'+' if real_adjustment > 0 else ''}{real_adjustment}% confidence adjustment"]

    mtf_pct, mtf_agree, mtf_total, mtf_detail = multi_timeframe_confirmation(yf_symbol, timeframe, raw_bias)
    if mtf_pct is not None:
        mtf_bonus = round((mtf_pct - 50) / 5)
        confidence = max(0, min(100, confidence + mtf_bonus))
        reasons = reasons + [f"multi-timeframe check: {mtf_agree}/{mtf_total} higher timeframes agree "
                              f"({'; '.join(mtf_detail)})"]

    rr = risk_reward(df, raw_bias, last)
    news_events = get_news_events()

    bias = raw_bias
    blocked_reasons = []
    if raw_bias != "NO TRADE / WAIT":
        passed, failures = passes_verification(score, rr["risk_reward_ratio"] if rr else None, news_events)
        if not passed:
            bias = "NO TRADE / WAIT"
            blocked_reasons = failures
            reasons = reasons + [f"⛔ {raw_bias} setup did not pass verification: {'; '.join(failures)}"]

    grade = setup_grade(confidence, rr["risk_reward_ratio"] if rr else None, mtf_pct, bias)
    breakdown = confidence_breakdown(last, winrate, trend, bias)

    tracker.log_signal(alias, timeframe, bias, confidence, score, current_price)

    change_pct = daily_pct_change(df)
    support, resistance = support_resistance(df)

    bias_key = (alias, timeframe)
    last_bias = _last_bias_by_key.get(bias_key)
    action = action_label(bias, last_bias)
    _last_bias_by_key[bias_key] = bias

    if news_events:
        soonest = news_events[0]
        reasons = reasons + [f"⚠ high-impact USD news: {soonest['title']} at "
                              f"{soonest['time'].astimezone().strftime('%-I:%M %p')}"]

    if action == "SELL":
        invalidation = [
            "This SELL means the earlier LONG's momentum faded (price/RSI/VWAP no longer "
            "all support it) — not a fresh short call.",
            "It would flip back to a LONG case if price reclaims VWAP and RSI climbs back over 55.",
        ]
    elif blocked_reasons:
        invalidation = [f"This setup is blocked, not invalidated — it didn't meet the bar: {f}"
                         for f in blocked_reasons]
    else:
        invalidation = invalidation_conditions(bias, last, rr)

    chart_data_url = None
    if include_chart:
        chart_png = render_candlestick_png(df, dark_mode=dark_mode, support=support, resistance=resistance)
        if chart_png:
            chart_data_url = "data:image/png;base64," + base64.b64encode(chart_png).decode("ascii")

    return {
        "symbol": alias,
        "name": DISPLAY_NAMES.get(yf_symbol, yf_symbol),
        "timeframe": timeframe,
        "bias": bias,
        "blocked_bias": raw_bias if blocked_reasons else None,
        "action": action,
        "confidence": confidence,
        "grade": grade,
        "confidence_breakdown": breakdown,
        "market_regime": regime,
        "mtf_agreement_pct": mtf_pct,
        "sample_size": sample_size,
        "price": current_price,
        "daily_change_pct": change_pct,
        "support": support,
        "resistance": resistance,
        "reasons": reasons,
        "invalidation": invalidation,
        "risk_reward": rr,
        "pnl_today": today_total(),
        "chart": chart_data_url,
    }


@app.route("/")
def index():
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(PUBLIC_DIR, filename)


@app.route("/api/signal")
def api_signal():
    raw_symbol = request.args.get("symbol", "ES")
    timeframe = request.args.get("timeframe", DEFAULT_TIMEFRAME)
    dark = request.args.get("dark", "1") == "1"
    result = build_signal(raw_symbol, timeframe=timeframe, include_chart=True, dark_mode=dark)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/signals")
def api_signals():
    symbols_param = request.args.get("symbols")
    dark = request.args.get("dark", "1") == "1"
    if symbols_param:
        symbols = [s.strip() for s in symbols_param.split(",") if s.strip()]
        pinned = []
    else:
        symbols, pinned = load_watchlist()
    results = []
    for sym in symbols:
        result = build_signal(sym, include_chart=False, dark_mode=dark)
        if "error" not in result:
            result["pinned"] = sym in pinned
            results.append(result)
    results.sort(key=lambda r: (not r["pinned"], -r["confidence"]))
    return jsonify(results)


@app.route("/api/watchlist", methods=["GET"])
def api_get_watchlist():
    symbols, pinned = load_watchlist()
    return jsonify({"symbols": symbols, "pinned": pinned})


@app.route("/api/watchlist/add", methods=["POST"])
def api_add_watchlist():
    payload = request.get_json(force=True)
    symbol = payload.get("symbol", "").strip()
    alias, yf_symbol = resolve_symbol_exact(symbol)
    if not yf_symbol:
        return jsonify({"error": f"Unrecognized symbol '{symbol}'"}), 400
    symbols, pinned = add_symbol(alias)
    return jsonify({"symbols": symbols, "pinned": pinned})


@app.route("/api/watchlist/remove", methods=["POST"])
def api_remove_watchlist():
    payload = request.get_json(force=True)
    symbols, pinned = remove_symbol(payload.get("symbol", ""))
    return jsonify({"symbols": symbols, "pinned": pinned})


@app.route("/api/watchlist/pin", methods=["POST"])
def api_pin_watchlist():
    payload = request.get_json(force=True)
    symbols, pinned = set_pinned(payload.get("symbol", ""), payload.get("pinned", True))
    return jsonify({"symbols": symbols, "pinned": pinned})


@app.route("/api/log_trade", methods=["POST"])
def api_log_trade():
    payload = request.get_json(force=True)
    try:
        pnl = log_trade(
            payload["symbol"], payload["side"].upper(),
            float(payload["contracts"]), float(payload["entry"]), float(payload["exit"]),
        )
        return jsonify({"pnl": pnl, "today_total": today_total()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/pnl")
def api_pnl():
    return jsonify(daily_totals())


@app.route("/api/timeframes")
def api_timeframes():
    return jsonify({"timeframes": list(TIMEFRAMES.keys()), "default": DEFAULT_TIMEFRAME})


@app.route("/api/performance")
def api_performance():
    return jsonify({
        "summary": tracker.recent_30_90_day_accuracy(),
        "by_symbol": tracker.performance_stats(group_by="symbol"),
        "by_timeframe": tracker.performance_stats(group_by="timeframe"),
    })


@app.route("/api/provider", methods=["GET", "POST"])
def api_provider():
    if request.method == "POST":
        payload = request.get_json(force=True)
        provider = payload.get("provider", "yahoo")
        if provider not in PROVIDERS:
            return jsonify({"error": f"Unknown provider '{provider}'"}), 400
        save_provider_config(provider, payload.get("api_key", ""))
        return jsonify({"ok": True})
    return jsonify({"current": load_provider_config(), "available": PROVIDERS})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
```
