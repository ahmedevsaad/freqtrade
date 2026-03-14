"""
HyroCandlePatternSessionStrategy — v9 Fixed
Freqtrade | Futures | Isolated Margin | 2x Leverage

ALL FIXES APPLIED
==================

CRITICAL LOOKAHEAD (4 fixes) — these make backtest results unreliable:
  [LB-1] sharp_move_lookback: was IntParameter used in populate_indicators.
         FT caches indicators once. Hyperopt epochs with lb != default(5)
         tested threshold against column built with lb=5. Now fixed int.
         Only sharp_move_threshold remains as optimizable Parameter.

  [LB-2] sideways_lookback: same issue. displacement column was cached
         with default=3. Now fixed int=3.

  [LB-3] close_change_threshold: was DecimalParameter used in
         populate_indicators to build pat_swing_short/pat_swing_long.
         Cached with default=0.003. Now fixed float.

  [LB-4] wick_ratio: was DecimalParameter used to build pat_shooting_star.
         Cached with default=3.2. Now fixed float.

MEDIUM LOOKAHEAD (1 fix):
  [LB-5] recent_high/recent_low: added .shift(1). Without it, rolling
         includes the current open (unfinished) candle on live.
         SL was based on partial candle data → backtest looked better
         than live reality.

CRITICAL BUG (1 fix):
  [BG-1] use_btc_ema_filter: removed entirely. Was CategoricalParameter
         but had no informative_pairs() and no gate in populate_entry_trend.
         Pure dead code wasting hyperopt budget.

MEDIUM BUGS (3 fixes):
  [BG-2] custom_sl_pct: space changed 'sell' → 'buy'.
         --spaces buy stoploss roi trailing never reaches 'sell'.
  [BG-3] custom_exit: both datetimes normalized to naive UTC before
         subtraction to prevent TypeError on tz mismatch.
  [BG-4] custom_exit: guard added for open_date_utc=None.

LOW BUGS (5 fixes):
  [BG-5] atr_pct: zero guard added on close (div/0 on data gaps).
  [BG-6] ADX atr_s: zero guard added (div/0 on identical first candle).
  [BG-7] bot_loop_start: Trade.get_trades() was called twice.
         active_ids now collected in main loop, reused for cleanup.
  [BG-8] _close_orphan_trade: current_time.timestamp() replaced with
         datetime.now(UTC).timestamp() to avoid local-tz assumption.
  [BG-9] All emojis removed from logger calls (Windows cp1252 fix).
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from freqtrade.persistence import Trade
from freqtrade.strategy import (CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy,
                                merge_informative_pair,
                                stoploss_from_absolute)
from pandas import DataFrame

logger = logging.getLogger(__name__)


class old(IStrategy):

    INTERFACE_VERSION = 3

    # ── Timeframe ─────────────────────────────────────────────────────────────
    timeframe                = "1h"
    # 1m detail: FT uses 1m candles to simulate precise entry/exit prices
    # inside each 1h candle. More realistic fills, especially for SL/trailing.
    # REQUIRES: freqtrade download-data --timeframes 1m 1h 4h --timerange ...
    # RAM WARNING: limit pairs to 20-30 for backtest (60x more data than no detail)
    # Live is fine — only current candle is fetched.
    timeframe_detail         = "1m"
    process_only_new_candles = True
    # startup_candle_count: covers warmup for all indicators.
    # EMA200(1h)=200h, EMA100(4h)=400h, ADX14=~100h.
    startup_candle_count     = 210
    can_short                = True

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    max_open_trades     = 10

    # ── ROI  (overridden by hyperopt --spaces roi) ────────────────────────────
    minimal_roi = {
        "0":    0.2,
        "414":  0.154,
        "1045": 0.078,
        "1691": 0,
    }

    # ── Stoploss  (overridden by hyperopt --spaces stoploss) ──────────────────
    # Hard floor — FT uses this when custom_stoploss returns None.
    # At 2x leverage: -0.107 price = -21.4% account per trade (hyperopt result).
    stoploss            = -0.107
    use_custom_stoploss = True

    # FIX [BG-2]: was space="sell" — unreachable with --spaces buy.
    custom_sl_pct = DecimalParameter(
        0.01, 0.10, default=0.09, decimals=3,
        space="buy", optimize=True, load=True,
    )

    # ── Trailing Stop  (overridden by hyperopt --spaces trailing) ─────────────
    trailing_stop                   = True
    trailing_stop_positive          = 0.011
    trailing_stop_positive_offset   = 0.087
    trailing_only_offset_is_reached = True

    # ── Time Stops ────────────────────────────────────────────────────────────
    time_stop_unprofitable: int = 16
    time_stop_hard:         int = 24

    # ── Fixed indicator periods ───────────────────────────────────────────────
    adx_period: int   = 14
    atr_period: int   = 14
    doji_ratio: float = 0.25

    # FIX [LB-3]: was DecimalParameter used in populate_indicators.
    # FT caches indicator columns — hyperopt cannot change these per-epoch.
    # Now fixed floats. To tune: run separate manual backtest comparisons.
    close_change_threshold: float = 0.003
    wick_ratio:             float = 3.2

    # ── ADX / DI Filter ───────────────────────────────────────────────────────
    use_adx_threshold: bool = True
    use_di_filter:     bool = True

    adx_threshold = DecimalParameter(
        15.0, 40.0, default=22.0, decimals=1,
        space="buy", optimize=True, load=True,
    )

    # ── Gate 6: Sharp Move Guard ──────────────────────────────────────────────
    # FIX [LB-1]: lookback was IntParameter used in populate_indicators.
    # Changing lookback changes the column shape — caching makes this
    # a critical lookahead bias source. Now fixed int.
    # Only the THRESHOLD is still hyperopt-optimizable (applied in entry).
    sharp_move_lookback:  int = 5   # FIXED — do not make this a Parameter

    sharp_move_threshold = DecimalParameter(
        1.5, 5.0, default=4.7, decimals=1,
        space="buy", optimize=True, load=True,
    )

    # ── Gate 7: Consecutive Candles Guard ─────────────────────────────────────
    max_consecutive_candles = IntParameter(
        2, 5, default=3,
        space="buy", optimize=True, load=True,
    )

    # ── Gate 8: Sideways / Ranging Market Filter ──────────────────────────────
    # FIX [LB-2]: sideways_lookback was IntParameter used in populate_indicators.
    # Same caching issue as sharp_move_lookback. Now fixed int.
    sideways_lookback: int = 3   # FIXED — do not make this a Parameter

    min_displacement = DecimalParameter(
        0.3, 2.0, default=0.5, decimals=1,
        space="buy", optimize=True, load=True,
    )

    # ── ATR Dynamic Stake Sizing ──────────────────────────────────────────────
    atr_high_threshold = DecimalParameter(
        2.0, 8.0, default=6.5, decimals=1,
        space="buy", optimize=True, load=True,
    )
    atr_mid_threshold = DecimalParameter(
        1.0, 4.0, default=1.4, decimals=1,
        space="buy", optimize=True, load=True,
    )
    atr_high_mult = DecimalParameter(
        0.2, 0.8, default=0.21, decimals=2,
        space="buy", optimize=True, load=True,
    )
    atr_mid_mult = DecimalParameter(
        0.5, 1.0, default=0.77, decimals=2,
        space="buy", optimize=True, load=True,
    )

    # ── Gate: Volatility Cap ─────────────────────────────────────────────────
    # Blocks ALL entries when pair ATR% > threshold.
    # ATR% = (ATR14 / close) * 100
    #   1-3%   = calm pair
    #   3-8%   = normal volatile
    #   8%+    = hyper-volatile meme coins like 1000RATS -> block
    max_atr_pct = DecimalParameter(
        3.0, 15.0, default=8.0, decimals=1,
        space="buy", optimize=True, load=True,
    )

    # ── Pattern Toggles ───────────────────────────────────────────────────────
    # Backtest analysis 2025 (full year):
    # bull_engulf:  trailing +21,313 BUT time_stop -27,922 = net LOSER  → OFF
    # morning_star: trailing  +3,326 BUT time_stop  -3,773 = net LOSER  → OFF
    # bear_engulf:  PF 1.43, strong trailing performance               → ON
    # swing_high:   PF 1.23, trailing PF 22.82                         → ON
    # swing_low:    PF 1.20, trailing PF 17.32                         → ON
    # shooting_star:PF 1.41, trailing PF 13.75                         → ON
    # evening_star: PF 1.11, marginal but positive                     → ON
    use_shooting_star: bool = True
    use_bear_engulf:   bool = True
    use_swing_high:    bool = True
    use_evening_star:  bool = True
    use_morning_star:  bool = True
    use_bull_engulf:   bool = True
    use_swing_low:     bool = True

    # ── Pair EMA Filter ───────────────────────────────────────────────────────
    # FIX [BG-1]: use_btc_ema_filter removed entirely — was dead code.
    # No informative_pairs() defined, no BTC columns created, no gate applied.
    use_pair_ema_filter = CategoricalParameter(
        [True, False], default=True,
        space="buy", optimize=True, load=True,
    )
    pair_ema_period: int = 200

    # ── 4h Trend Filter ───────────────────────────────────────────────────────
    # EMA100(4h) = 100 * 4h = 400h horizon — genuinely different from EMA200(1h).
    # EMA50(4h)  = 50  * 4h = 200h = same as EMA200(1h) → redundant.
    # EMA100(4h) = 400h = ~17 days trend = medium-term regime filter.
    # Long  only when 4h close > EMA100(4h) — HTF bullish
    # Short only when 4h close < EMA100(4h) — HTF bearish
    use_4h_ema_filter = CategoricalParameter(
        [True, False], default=True,
        space="buy", optimize=True, load=True,
    )
    ema_4h_period: int = 100  # EMA100 on 4h = 400h = 17 day trend (distinct from EMA200 1h)

    # ── Round Number Protection ───────────────────────────────────────────────
    # Very strong psychological levels act as hard support/resistance.
    # Near these levels price is unpredictable — bounces or rejections are
    # more likely than clean trend continuation.
    #
    # "Very strong" = price-independent: 10^n and 5*10^n and 2*10^n
    # Examples: ..., 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500,
    #           1000, 2000, 5000, 10000, 20000, 50000, 100000 ...
    #
    # Logic:
    #   Near round number → price could BOUNCE up  → block SHORT entry
    #   Near round number → price could REJECT down → block LONG  entry
    #   (Both directions blocked — the level is a magnet, not a bias)
    #
    # Proximity threshold = % distance from the round level.
    # 0.5% = tight (only right at the level)
    # 1.0% = wider (1% radius around the level)
    use_round_number_filter = CategoricalParameter(
        [True, False], default=True,
        space="buy", optimize=True, load=True,
    )
    round_number_pct = DecimalParameter(
        0.3, 2.0, default=1.0, decimals=1,
        space="buy", optimize=True, load=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  INFORMATIVE PAIRS
    # ══════════════════════════════════════════════════════════════════════════

    def informative_pairs(self):
        # Fetch 4h candles for every pair in the whitelist
        # FT downloads this data automatically during backtest/live
        pairs = self.dp.current_whitelist()
        return [(pair, "4h") for pair in pairs]

    # ══════════════════════════════════════════════════════════════════════════
    #  HOOKS
    # ══════════════════════════════════════════════════════════════════════════

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._sl_sent: dict = {}

    def leverage(self, pair, current_time, current_rate,
                 proposed_leverage, max_leverage, entry_tag, side, **kwargs):
        return min(2.0, max_leverage)

    @property
    def plot_config(self):
        return {
            "main_plot": {
                # Trend filters
                "pair_ema200":   {"color": "#f0c040", "type": "line"},
                "ema100_4h_4h": {"color": "#00aaff", "type": "line"},
                # Round number levels — 3 strongest levels around current price
                # Nearest = brightest red, 2nd/3rd = progressively dimmer
                # "rn_level_1": {"color": "#ff2222", "type": "line"},
                # "rn_level_2": {"color": "#ff6666", "type": "line"},
                # "rn_level_3": {"color": "#ffaaaa", "type": "line"},
            },
            "subplots": {
                "ADX / DI": {
                    "adx14":    {"color": "#f0c040", "type": "line"},
                    "plus_di":  {"color": "#00c896", "type": "line"},
                    "minus_di": {"color": "#ff4f6e", "type": "line"},
                },
                "Volume": {
                    "volume":      {"color": "#5588ff", "type": "bar"},
                    "volume_ma20": {"color": "#ffaa00", "type": "line"},
                },
                "Volatility (ATR%)": {
                    "atr_pct":      {"color": "#ff8800", "type": "line"},
                    "displacement": {"color": "#00e5ff", "type": "line"},
                },
                "Round Proximity %": {
                    "_rn_dist": {"color": "#ff4444", "type": "line"},
                },
            },
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  STAKE SIZING
    # ══════════════════════════════════════════════════════════════════════════

    def custom_stake_amount(self, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, **kwargs) -> float:
        pair = kwargs.get("pair", "")
        if not pair:
            return proposed_stake

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return proposed_stake

        atr_pct = dataframe["atr_pct"].iat[-1]
        if pd.isna(atr_pct) or atr_pct <= 0:
            return proposed_stake

        high_thresh = max(self.atr_high_threshold.value, self.atr_mid_threshold.value)
        mid_thresh  = min(self.atr_high_threshold.value, self.atr_mid_threshold.value)

        # FIX [BG-9]: emojis removed — Windows cp1252 crash in hyperopt workers
        if atr_pct > high_thresh:
            mult = self.atr_high_mult.value
            logger.debug("[VOL-HIGH] %s atr=%.1f%% -> stake x%.2f", pair, atr_pct, mult)
        elif atr_pct > mid_thresh:
            mult = self.atr_mid_mult.value
            logger.debug("[VOL-MID] %s atr=%.1f%% -> stake x%.2f", pair, atr_pct, mult)
        else:
            mult = 1.0

        adjusted = proposed_stake * mult
        if min_stake is not None and adjusted < min_stake:
            adjusted = min_stake
        if max_stake is not None and adjusted > max_stake:
            adjusted = max_stake
        return adjusted

    # ══════════════════════════════════════════════════════════════════════════
    #  BYBIT LIVE LOOP
    # ══════════════════════════════════════════════════════════════════════════

    def bot_loop_start(self, current_time, **kwargs) -> None:
        if self.config.get("dry_run", True):
            return

        # FIX [BG-7]: was calling Trade.get_trades() twice.
        # active_ids collected during main loop, reused for stale-cache cleanup.
        active_ids: set = set()

        for t in Trade.get_trades(trade_filter=Trade.is_open.is_(True)).all():
            sym = t.pair.split(":")[0].replace("/", "")
            active_ids.add(t.id)

            try:
                pos   = self.dp._exchange._api.privateGetV5PositionList(
                    {"category": "linear", "symbol": sym})
                plist = pos.get("result", {}).get("list", [])
                size  = float(plist[0]["size"]) if plist else 0.0

                if size == 0:
                    self._close_orphan_trade(t, sym, current_time)
                    continue
                if size < t.amount * 0.99:
                    logger.warning("[WARN] Partial close #%d %s: Bybit=%.8f FT=%.8f",
                                   t.id, sym, size, t.amount)
            except Exception as e:
                logger.debug("Position check skip %s: %s", sym, e)
                continue

            sl = t.stop_loss
            if not sl or sl <= 0 or self._sl_sent.get(t.id) == sl:
                continue

            try:
                resp = self.dp._exchange._api.privatePostV5PositionTradingStop({
                    "category":    "linear",
                    "symbol":      sym,
                    "stopLoss":    str(round(sl, 8)),
                    "slTriggerBy": "MarkPrice",
                    "tpslMode":    "Full",
                    "positionIdx": 0,
                })
                code = resp.get("retCode", -1)
                if code in (0, 110043, 34040):
                    self._sl_sent[t.id] = sl
                    logger.info("[SL-SYNC] %s %s sl=%.6f",
                                sym, "Short" if t.is_short else "Long", sl)
                else:
                    logger.warning("[WARN] SL failed %s: code=%s msg=%s",
                                   sym, code, resp.get("retMsg"))
            except Exception as e:
                if "not modified" in str(e).lower() or "34040" in str(e):
                    self._sl_sent[t.id] = sl
                else:
                    logger.warning("[WARN] SL failed %s: %s", sym, e)

        for k in [k for k in list(self._sl_sent) if k not in active_ids]:
            del self._sl_sent[k]

    def _close_orphan_trade(self, t, sym: str, current_time) -> None:
        close_rate = t.stop_loss if t.stop_loss else t.open_rate
        close_pnl  = None
        close_time = None

        try:
            cpnl      = self.dp._exchange._api.privateGetV5PositionClosedPnl(
                {"category": "linear", "symbol": sym, "limit": "1"})
            cpnl_list = cpnl.get("result", {}).get("list", [])
            if cpnl_list:
                rec        = cpnl_list[0]
                updated_ms = int(rec.get("updatedTime", "0"))
                # FIX [BG-8]: was current_time.timestamp() which uses LOCAL tz.
                # On non-UTC servers age_s is off by hours → window never hit.
                now_ts = datetime.now(tz=timezone.utc).timestamp()
                age_s  = now_ts - updated_ms / 1000

                if 0 <= age_s < 300:
                    close_rate = float(rec.get("avgExitPrice", close_rate))
                    close_pnl  = float(rec.get("closedPnl", 0))
                    close_time = datetime.fromtimestamp(
                        updated_ms / 1000, tz=timezone.utc)
                else:
                    logger.debug("Closed PnL too old for %s (%.0fs ago), "
                                 "using fallback rate", sym, age_s)
        except Exception as ex:
            logger.debug("Closed PnL fetch skip %s: %s", sym, ex)

        logger.info("[ORPHAN] Bybit closed #%d %s @ %.8f", t.id, sym, close_rate)
        t.close(close_rate)

        if close_pnl is not None:
            t.close_profit_abs = close_pnl
            t.realized_profit  = close_pnl
            if t.open_trade_value:
                t.close_profit = (close_pnl / t.open_trade_value) * (t.leverage or 1)
        else:
            prof               = t.calculate_profit(close_rate)
            t.close_profit     = prof.profit_ratio
            t.close_profit_abs = prof.profit_abs
            t.realized_profit  = prof.profit_abs

        if close_time:
            t.close_date = close_time

        Trade.commit()
        self._sl_sent.pop(t.id, None)

    # ══════════════════════════════════════════════════════════════════════════
    #  INDICATORS
    # ══════════════════════════════════════════════════════════════════════════

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe["volume_ma20"] = dataframe["volume"].rolling(20).mean()

        # ── 4h EMA Trend Filter ───────────────────────────────────────────────
        # Fetch 4h candles for this pair, compute EMA50, merge into 1h frame.
        # merge_informative_pair forward-fills the 4h value into each 1h candle
        # that belongs to it — NO lookahead (4h candle only merges after close).
        #
        # FIX: column names avoid numeric prefix ("4h_bull" -> "trend_4h_bull")
        #      numeric-prefix names are valid in pandas bracket notation but
        #      cause silent failures in some FT versions and attr access.
        # FIX: explicit .fillna(False).astype(bool) after merge to convert
        #      NaN rows (startup period before first 4h candle) to False,
        #      avoiding object-dtype columns and ambiguous NaN comparisons.
        informative_4h = self.dp.get_pair_dataframe(
            pair=metadata["pair"], timeframe="4h"
        )
        if not informative_4h.empty:
            ema_4h = (
                informative_4h["close"]
                .ewm(span=self.ema_4h_period, adjust=False)
                .mean()
            )
            informative_4h["ema100_4h"]    = ema_4h
            # Use clean names — no numeric prefix to avoid pandas edge cases
            informative_4h["trend_4h_bull"] = informative_4h["close"] > ema_4h
            informative_4h["trend_4h_bear"] = informative_4h["close"] < ema_4h

            dataframe = merge_informative_pair(
                dataframe, informative_4h,
                self.timeframe, "4h",
                ffill=True,
            )
            # After merge, FT appends "_4h" suffix to all informative columns.
            # trend_4h_bull -> trend_4h_bull_4h  |  NaN in startup rows.
            # fillna(False): block entries when 4h data not yet available (safe).
            # astype(bool): ensure bool dtype, avoid object dtype slowness.
            dataframe["trend_4h_bull"] = (
                dataframe["trend_4h_bull_4h"]
                .infer_objects(copy=False)
                .fillna(False)
                .astype(bool)
            )
            dataframe["trend_4h_bear"] = (
                dataframe["trend_4h_bear_4h"]
                .infer_objects(copy=False)
                .fillna(False)
                .astype(bool)
            )
        else:
            # No 4h data available (pair just listed, or data not downloaded).
            # Default to True = allow all entries (filter disabled gracefully).
            # Will print a warning so the user knows to download 4h data.
            logger.warning("[4H-FILTER] No 4h data for %s -- filter disabled",
                           metadata.get("pair", "?"))
            dataframe["trend_4h_bull"] = True
            dataframe["trend_4h_bear"] = True

        # ── Pair EMA ──────────────────────────────────────────────────────────
        pair_ema = dataframe["close"].ewm(span=self.pair_ema_period, adjust=False).mean()
        dataframe["pair_ema200"]   = pair_ema
        dataframe["pair_ema_bull"] = dataframe["close"] > pair_ema
        dataframe["pair_ema_bear"] = dataframe["close"] < pair_ema

        # ── Round Number Proximity ────────────────────────────────────────────
        # Detects very strong psychological price levels independent of scale.
        # Level set: 10^n, 2*10^n, 5*10^n  (e.g. 1, 2, 5, 10, 20, 50, 100 ...)
        # Computed per-candle using close price magnitude.
        # Result: near_round_number = True when close is within round_number_pct%
        # of any such level. Used in entry gate to block both long and short.
        close = dataframe["close"]
        # magnitude = largest power of 10 below close (e.g. close=4700 → mag=1000)
        # FIX: use 10.0 (float) not 10 (int) — negative exponents (price < 1) cause
        # "Integers to negative integer powers are not allowed" with int base.
        magnitude = (10.0 ** (close.apply(
            lambda x: int(__import__('math').floor(__import__('math').log10(x)))
            if x > 0 else 0
        ))).astype(float)
        # The 6 candidate strong round levels around any price
        cand1 = magnitude          # e.g. 1000
        cand2 = 2.0 * magnitude    # e.g. 2000
        cand3 = 5.0 * magnitude    # e.g. 5000
        cand4 = 10.0 * magnitude   # e.g. 10000
        cand5 = magnitude / 2.0    # e.g. 500
        cand6 = magnitude / 5.0    # e.g. 200
        # near = within pct% of any candidate (fixed float, not Parameter — see note)
        # NOTE: round_number_pct is used in populate_entry_trend at gate time
        # because it only applies a threshold, not a column shape change — safe.
        def _near(candidate):
            return ((close - candidate).abs() / candidate)
        dist_df = pd.concat([
            _near(cand1), _near(cand2), _near(cand3),
            _near(cand4), _near(cand5), _near(cand6),
        ], axis=1)
        levels_df = pd.concat([cand1, cand2, cand3, cand4, cand5, cand6], axis=1)
        # Index of nearest level per row
        nearest_idx = dist_df.values.argmin(axis=1)
        dataframe["_rn_dist"]  = dist_df.values.min(axis=1)

        # ── Round Level Lines (for plotting) ─────────────────────────────────
        # Plot the 3 strongest levels around current price as price lines.
        # Sorted by distance so rn_level_1 = nearest, rn_level_2 = 2nd, etc.
        # These are price-scale lines that render as horizontal-ish bands on chart.
        sorted_idx = dist_df.values.argsort(axis=1)
        lvl_arr = levels_df.values
        dataframe["rn_level_1"] = lvl_arr[range(len(lvl_arr)), sorted_idx[:, 0]].astype(float)
        dataframe["rn_level_2"] = lvl_arr[range(len(lvl_arr)), sorted_idx[:, 1]].astype(float)
        dataframe["rn_level_3"] = lvl_arr[range(len(lvl_arr)), sorted_idx[:, 2]].astype(float)
        # Store the raw distance — threshold applied in entry trend (hyperopt-safe)

        # ── Recent High / Low ─────────────────────────────────────────────────
        # FIX [LB-5]: .shift(1) added. rolling without shift includes the
        # current open candle on live → SL based on partial data → backtest
        # SL is wider than what you actually get live.
        dataframe["recent_high"] = dataframe["high"].rolling(14).max().shift(1)
        dataframe["recent_low"]  = dataframe["low"].rolling(14).min().shift(1)

        # ── ATR ───────────────────────────────────────────────────────────────
        hl  = dataframe["high"] - dataframe["low"]
        hpc = (dataframe["high"] - dataframe["close"].shift(1)).abs()
        lpc = (dataframe["low"]  - dataframe["close"].shift(1)).abs()
        tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)

        dataframe["atr14"] = tr.ewm(alpha=1.0 / self.atr_period, adjust=False).mean()
        # FIX [BG-5]: zero guard on close to prevent inf in atr_pct
        safe_close = dataframe["close"].replace(0, float("nan"))
        dataframe["atr_pct"] = (dataframe["atr14"] / safe_close) * 100

        # ── ADX / DI ──────────────────────────────────────────────────────────
        alpha    = 1.0 / self.adx_period
        h_diff   = dataframe["high"].diff()
        l_diff   = dataframe["low"].diff().mul(-1)
        plus_dm  = h_diff.where((h_diff > l_diff) & (h_diff > 0), 0.0)
        minus_dm = l_diff.where((l_diff > h_diff) & (l_diff > 0), 0.0)
        atr_s    = tr.ewm(alpha=alpha, adjust=False).mean()
        # FIX [BG-6]: zero guard on atr_s (div/0 if first candle TR=0)
        safe_atr_s = atr_s.replace(0, float("nan"))
        plus_di  = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / safe_atr_s
        minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / safe_atr_s
        dataframe["plus_di"]  = plus_di
        dataframe["minus_di"] = minus_di
        di_sum = plus_di + minus_di
        dx     = (100 * (plus_di - minus_di).abs() / di_sum).where(di_sum > 0, 0.0)
        dataframe["adx14"] = dx.ewm(alpha=alpha, adjust=False).mean()

        # ── Gate 6: Sharp Move Guard ──────────────────────────────────────────
        # FIX [LB-1]: lookback is now a FIXED int (was IntParameter).
        # FT caches indicator columns once per backtest/hyperopt run.
        # IntParameter here caused hyperopt to test lb=2,3,4 against
        # a column that was always built with lb=5 (default).
        # The THRESHOLD is still optimizable — applied in populate_entry_trend.
        safe_atr14 = dataframe["atr14"].replace(0, 1e-10)
        lb = self.sharp_move_lookback   # fixed int=5
        dataframe["sharp_drop"]  = (
            (dataframe["close"].shift(lb) - dataframe["close"]) / safe_atr14
        )
        dataframe["sharp_spike"] = (
            (dataframe["close"] - dataframe["close"].shift(lb)) / safe_atr14
        )

        # ── Gate 7: Consecutive Candles Guard ─────────────────────────────────
        # BUG-6 FIX: original used (c < o) which resets count on doji (c==o).
        # A doji mid-run (3bear-doji-2bear) reset consec_bear to 0 making the
        # guard ineffective for split bear runs. Fix: treat doji as continuation
        # of the previous candle direction using forward-fill.
        c = dataframe["close"]
        o = dataframe["open"]
        # 1=bear, -1=bull, 0=doji
        direction = pd.Series(0, index=dataframe.index)
        direction[c < o] = 1   # bear
        direction[c > o] = -1  # bull
        # Forward-fill doji with previous direction (neutral doji = continuation)
        direction = direction.replace(0, float("nan")).ffill().fillna(0).astype(int)
        is_bear_candle = (direction == 1).astype(int)
        is_bull_candle = (direction == -1).astype(int)
        bear_group = (is_bear_candle != is_bear_candle.shift(1)).cumsum()
        bull_group = (is_bull_candle != is_bull_candle.shift(1)).cumsum()
        # BUG-D FIX: groupby(Series) emits FutureWarning in pandas 2.x.
        # Pass .values (ndarray) instead to silence warning and future-proof.
        dataframe["consec_bear"] = is_bear_candle.groupby(bear_group.values).cumsum()
        dataframe["consec_bull"] = is_bull_candle.groupby(bull_group.values).cumsum()

        # ── Gate 8: Sideways / Ranging Market Filter ──────────────────────────
        # FIX [LB-2]: lookback is now a FIXED int (was IntParameter).
        # Same caching issue as sharp_move_lookback.
        sl = self.sideways_lookback   # fixed int=3
        dataframe["displacement"] = (
            (dataframe["close"] - dataframe["close"].shift(sl)).abs() / safe_atr14
        )

        # ── Candlestick Parts ─────────────────────────────────────────────────
        # FIX [LB-3, LB-4]: close_change_threshold and wick_ratio are now
        # FIXED FLOATS (were DecimalParameters).
        # Using Parameter.value here to build pattern columns creates the
        # same caching problem — the column is built once with default values
        # and all hyperopt epochs test against that stale column.
        h = dataframe["high"]
        l = dataframe["low"]
        body         = (c - o).abs()
        candle_range = h - l
        upper_wick   = h - c.where(c > o, o)
        lower_wick   = c.where(c < o, o) - l
        is_bull      = c > o
        is_bear      = c < o
        # BUG-H FIX: candle_range.replace(0, 1) substitutes 1 USDT for frozen-OHLC rows.
        # For BTC@90k this makes body < 0.25 USDT impossible (never a doji).
        # For 0.001-coins body < 0.25 USDT is always True (always a doji).
        # Fix: replace(0, nan) so zero-range candles evaluate to NaN (excluded from is_small).
        is_small = body < (self.doji_ratio * candle_range.replace(0, float("nan")))

        cct = self.close_change_threshold   # fixed float
        wr  = self.wick_ratio               # fixed float

        # ── Patterns ──────────────────────────────────────────────────────────
        raw_swing_short = (
            (h < h.shift(1)) & (h.shift(1) > h.shift(2))
            & ((c.shift(1) - c) / c.shift(1).replace(0, 1) > cct)
        )
        # BUG-13 FIX: .astype(bool) converts NaN → True (numpy: bool(nan)=True).
        # First ~3 rows have NaN from .shift(1)/.shift(2) and would fire falsely.
        # .fillna(False) first ensures NaN rows become False, not True.
        dataframe["pat_swing_short"] = raw_swing_short.shift(1).fillna(False).astype(bool) & is_bear

        dataframe["pat_shooting_star"] = (
            # BUG-3 FIX: added (body > 0) guard.
            # When body=0 (doji), body.replace(0, 0.0001) makes the wick
            # ratio condition trivially True for any crypto price > 1 USDT.
            # A gravestone doji is a separate pattern -- exclude body=0 here
            # to keep shooting_star semantically clean.
            # BUG-12 FIX: added is_bear guard.
            # Without it, bullish inverted hammers (large upper wick, small body, c>o)
            # also match -- triggering SHORT entries on a BULLISH candle. Wrong.
            # Shooting star is a bearish reversal: candle must close below open.
            is_bear
            & (body > 0)
            & (upper_wick >= wr * body)
            & (lower_wick <= 0.5 * body)
            & (candle_range > 0)
        )
        dataframe["pat_bear_engulf"] = (
            is_bear & is_bull.shift(1)
            & (o >= c.shift(1)) & (c <= o.shift(1))
        )
        dataframe["pat_evening_star"] = (
            is_bull.shift(2) & is_small.shift(1) & is_bear
            & (c < (o.shift(2) + c.shift(2)) / 2)
        )

        raw_swing_long = (
            (l > l.shift(1)) & (l.shift(1) < l.shift(2))
            & ((c - c.shift(1)) / c.shift(1).replace(0, 1) > cct)
        )
        dataframe["pat_swing_long"] = raw_swing_long.shift(1).fillna(False).astype(bool) & is_bull

        dataframe["pat_bull_engulf"] = (
            is_bull & is_bear.shift(1)
            & (o <= c.shift(1)) & (c >= o.shift(1))
        )
        dataframe["pat_morning_star"] = (
            is_bear.shift(2) & is_small.shift(1) & is_bull
            & (c > (o.shift(2) + c.shift(2)) / 2)
        )

        return dataframe

    # ══════════════════════════════════════════════════════════════════════════
    #  ENTRY SIGNAL
    # ══════════════════════════════════════════════════════════════════════════

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe["enter_long"]  = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"]   = ""

        # BUG-A FIX: Python ternary `x if cond else y` has LOWEST precedence.
        # Old: `a & b & (c if cond else d)` parses as `(a & b & c) if cond else d`
        # When use_adx=False entire common = pd.Series(True) -- volume filter bypassed!
        # Fix: compute ADX condition separately, then combine with &.
        if self.use_adx_threshold:
            adx_ok = dataframe["adx14"] > self.adx_threshold.value
        else:
            adx_ok = pd.Series(True, index=dataframe.index)

        common = (
            (dataframe["volume"] > 0)
            & (dataframe["volume"] > dataframe["volume_ma20"])
            & adx_ok
        )

        if self.use_di_filter:
            di_long_ok  = dataframe["plus_di"]  > dataframe["minus_di"]
            di_short_ok = dataframe["minus_di"] > dataframe["plus_di"]
        else:
            di_long_ok  = pd.Series(True, index=dataframe.index)
            di_short_ok = pd.Series(True, index=dataframe.index)

        if self.use_pair_ema_filter.value:
            pair_long_ok  = dataframe["pair_ema_bull"]
            pair_short_ok = dataframe["pair_ema_bear"]
        else:
            pair_long_ok  = pd.Series(True, index=dataframe.index)
            pair_short_ok = pd.Series(True, index=dataframe.index)

        # ── Gate: 4h Trend Filter ─────────────────────────────────────────────
        # Long  only if 4h close > EMA50(4h)
        # Short only if 4h close < EMA50(4h)
        # This single gate cuts most counter-trend entries that end up as
        # time_stop_unprofitable losses.
        if self.use_4h_ema_filter.value:
            trend_4h_long_ok  = dataframe["trend_4h_bull"]
            trend_4h_short_ok = dataframe["trend_4h_bear"]
        else:
            trend_4h_long_ok  = pd.Series(True, index=dataframe.index)
            trend_4h_short_ok = pd.Series(True, index=dataframe.index)

        # Thresholds evaluated HERE — correct for hyperopt
        thr = self.sharp_move_threshold.value
        no_short_chase = (
            (dataframe["sharp_drop"]  <= thr)
            & (dataframe["consec_bear"] < self.max_consecutive_candles.value)
        )
        no_long_chase = (
            (dataframe["sharp_spike"] <= thr)
            & (dataframe["consec_bull"] < self.max_consecutive_candles.value)
        )

        market_trending = dataframe["displacement"] >= self.min_displacement.value

        # Gate: Volatility Cap — blocks hyper-volatile pairs (e.g. 1000RATS)
        # max_atr_pct=0.0 disables the filter entirely
        if self.max_atr_pct.value > 0:
            not_hyper_volatile = dataframe["atr_pct"] <= self.max_atr_pct.value
        else:
            not_hyper_volatile = pd.Series(True, index=dataframe.index)

        # ── Gate: Round Number Protection ────────────────────────────────────
        # Near 10^n / 2*10^n / 5*10^n → price unpredictable → block both sides.
        # _rn_dist = fractional distance to nearest strong round level.
        # Threshold from hyperopt (round_number_pct / 100).
        if self.use_round_number_filter.value:
            not_near_round = (
                dataframe["_rn_dist"] > self.round_number_pct.value / 100.0
            )
        else:
            not_near_round = pd.Series(True, index=dataframe.index)

        long_ok = (
            common & di_long_ok & pair_long_ok
            & trend_4h_long_ok
            & no_long_chase & market_trending
            & not_hyper_volatile
            & not_near_round
        )
        short_ok = (
            common & di_short_ok & pair_short_ok
            & trend_4h_short_ok
            & no_short_chase & market_trending
            & not_hyper_volatile
            & not_near_round
        )

        if self.use_morning_star:
            dataframe.loc[dataframe["pat_morning_star"] & long_ok,  ["enter_long",  "enter_tag"]] = [1, "morning_star"]
        if self.use_bull_engulf:
            dataframe.loc[dataframe["pat_bull_engulf"]  & long_ok,  ["enter_long",  "enter_tag"]] = [1, "bull_engulf"]
        if self.use_swing_low:
            dataframe.loc[dataframe["pat_swing_long"]   & long_ok,  ["enter_long",  "enter_tag"]] = [1, "swing_low"]
        if self.use_evening_star:
            dataframe.loc[dataframe["pat_evening_star"] & short_ok, ["enter_short", "enter_tag"]] = [1, "evening_star"]
        if self.use_bear_engulf:
            dataframe.loc[dataframe["pat_bear_engulf"]  & short_ok, ["enter_short", "enter_tag"]] = [1, "bear_engulf"]
        if self.use_shooting_star:
            dataframe.loc[dataframe["pat_shooting_star"]& short_ok, ["enter_short", "enter_tag"]] = [1, "shooting_star"]
        if self.use_swing_high:
            dataframe.loc[dataframe["pat_swing_short"]  & short_ok, ["enter_short", "enter_tag"]] = [1, "swing_high"]

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, "exit_long"]  = 0
        dataframe.loc[:, "exit_short"] = 0
        return dataframe

    # ══════════════════════════════════════════════════════════════════════════
    #  CUSTOM STOPLOSS
    # ══════════════════════════════════════════════════════════════════════════

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float,
                        after_fill: bool, **kwargs) -> Optional[float]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None

        last        = dataframe.iloc[-1].squeeze()
        # FIX [LB-5]: .shift(1) applied in populate_indicators.
        # recent_high/recent_low reflect only CLOSED candles.
        # FIX [BG-10]: last.get() returns NaN when the key exists but the value
        # is NaN (rolling(14).shift(1) produces NaN for the first 14 candles).
        # stoploss_from_absolute(NaN, ...) returns NaN which FT treats as
        # "no custom SL" and falls back to main stoploss — but explicit guard
        # is safer and avoids silent misbehaviour on brand-new pairs.
        recent_high_raw = last.get("recent_high", current_rate)
        recent_low_raw  = last.get("recent_low",  current_rate)
        recent_high = current_rate if pd.isna(recent_high_raw) else float(recent_high_raw)
        recent_low  = current_rate if pd.isna(recent_low_raw)  else float(recent_low_raw)
        sl_pct      = float(self.custom_sl_pct.value)

        if trade.is_short:
            sl_absolute = recent_high + (current_rate * sl_pct)
            # BUG-F FIX: In fast-rising markets, recent_high (14-candle shifted max)
            # can be BELOW current_rate. SL below current price for a short = wrong.
            # Guard: SL must always be ABOVE current price for shorts.
            sl_absolute = max(sl_absolute, current_rate * 1.001)
            return stoploss_from_absolute(
                sl_absolute, current_rate, is_short=True, leverage=trade.leverage)
        else:
            sl_absolute = recent_low - (current_rate * sl_pct)
            # BUG-G FIX: In fast-falling markets, recent_low (14-candle shifted min)
            # can be ABOVE current_rate. SL above current price for a long = wrong.
            # Guard: SL must always be BELOW current price for longs.
            sl_absolute = min(sl_absolute, current_rate * 0.999)
            return stoploss_from_absolute(
                sl_absolute, current_rate, is_short=False, leverage=trade.leverage)

    # ══════════════════════════════════════════════════════════════════════════
    #  TIME-BASED EXITS
    # ══════════════════════════════════════════════════════════════════════════

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float,
                    **kwargs) -> Optional[str]:
        # FIX [BG-4]: guard against None open_date_utc
        if not trade.open_date_utc or not current_time:
            return None

        # FIX [BG-3]: normalize both to naive UTC before subtraction.
        # If one is tz-aware and other is naive -> TypeError crash.
        ct      = current_time.replace(tzinfo=None) if current_time.tzinfo else current_time
        open_dt = trade.open_date_utc.replace(tzinfo=None)
        hours   = int((ct - open_dt).total_seconds() / 3600)

        if hours >= self.time_stop_hard:
            return "time_stop_hard"
        if hours >= self.time_stop_unprofitable and current_profit <= 0:
            return "time_stop_unprofitable"
        return None