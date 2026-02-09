"""Trading Bot Engine
Handles all trading logic, signal processing, and order execution.
Uses structured logging with tags for easy troubleshooting.
"""
import asyncio
from datetime import datetime, timezone, timedelta
import logging
import random

from config import bot_state, config, DB_PATH
from indices import get_index_config, round_to_strike
from utils import get_ist_time, is_market_open, can_take_new_trade, should_force_squareoff, format_timeframe
from indicators import SuperTrend, MACD, ADX
from score_engine import ScoreEngine, Candle
from position_sizing import PositionSizingAgent
from dhan_api import DhanAPI
from database import save_trade, update_trade_exit

# NEW: Comprehensive Trading Logic v2
from market_regime_v2 import MarketRegimeDetector, MarketRegime
from confidence_v2 import ConfidenceCalculator, ConfidenceLevel, ConfidenceBreakdown
from entry_logic_v2 import EntryLogic, EntryDirection, EntryConfirmation, EntryDecision
from exit_logic_v2 import ExitLogic, ExitReason, ExitSignal
from cooldown_v2 import CooldownManager, CooldownReason, CooldownState

logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot engine"""
    
    def __init__(self):
        self.running = False
        self.task = None
        self.dhan = None
        self.current_position = None
        self.entry_price = 0.0
        self.trailing_sl = None
        self.highest_profit = 0.0
        self.indicator = None  # Will hold selected indicator
        self.htf_indicator = None  # Higher-timeframe SuperTrend (e.g., 1m filter)
        self.macd = None  # LTF MACD for confirmation
        self.adx = None  # Optional ADX(14) for regime detection
        self.score_engine = None  # Multi-timeframe score engine (optional)
        self._mds_last_direction = None
        self._mds_confirm_count = 0
        self.last_exit_candle_time = None
        self.last_trade_time = None  # For min_trade_gap protection
        self.last_signal = None  # For trade_only_on_flip protection
        self._last_entries_paused_log_time = None
        self.entry_time_utc = None  # datetime for min-hold exit protection
        self.last_order_time_utc = None  # datetime for order cooldown (entry/exit pacing)
        self._paper_replay_candles = []
        self._paper_replay_pos = 0
        self._paper_replay_htf_elapsed = 0
        self._last_mds_candle_ts = None
        self._mds_htf_count = 0
        self._mds_htf_high = 0.0
        self._mds_htf_low = float('inf')
        self._mds_htf_close = 0.0
        self._initialize_indicator()

        # ═══════════════════════════════════════════════════════════════════
        # NEW: Comprehensive Trading Logic v2 Components
        # ═══════════════════════════════════════════════════════════════════
        
        # Market Regime Detection (Gatekeeper)
        self.regime_detector = MarketRegimeDetector(
            use_adx=bool(config.get('use_adx_for_regime', False)),
            adx_threshold=float(config.get('adx_threshold', 20.0))
        )
        
        # Confidence Scoring Engine
        self.confidence_calc = ConfidenceCalculator()
        
        # Entry Decision Logic
        self.entry_logic = EntryLogic()
        
        # Exit Decision Logic (Risk First)
        self.exit_logic = ExitLogic()
        
        # Cooldown Manager (Anti-Chop)
        self.cooldown_mgr = CooldownManager(
            cooldown_candles=int(config.get('cooldown_candles', 3))
        )
        
        # Runtime confidence tracking during position hold
        self.runtime_confidence = 100.0
        
        # Entry confidence recorded at time of entry
        self.entry_confidence_recorded = 0.0
        
        # Track previous ST direction for flip detection
        self.prev_st_direction = None
        
        # Track candles held in position
        self.candles_in_position = 0
        
        # Track previous MACD histogram for trend detection
        self.prev_macd_histogram = None

        # Track recent candle ranges for volatility expansion scoring
        self._recent_candle_ranges = []
        
        logger.info("[INIT] Trading Logic v2 initialized")
        logger.info(f"[CONFIG] Regime: {self.regime_detector.use_adx and 'ADX' or 'ST Distance'} | Confidence Entry: {ConfidenceCalculator.MIN_ENTRY_CONFIDENCE} | Confidence Exit: {ConfidenceCalculator.MIN_RUNTIME_CONFIDENCE} | Cooldown: {self.cooldown_mgr.cooldown_duration} candles")

    def _prefetch_candles_needed(self) -> int:
        st_period = int(config.get('supertrend_period', 7) or 7)
        macd_slow = int(config.get('macd_slow', 26) or 26)
        macd_signal = int(config.get('macd_signal', 9) or 9)

        # SuperTrend needs at least `period` candles; MACD needs slow EMA + signal EMA.
        base_needed = max(st_period + 1, macd_slow + macd_signal)

        # If HTF filter is enabled (fixed to 60s in current code), seed enough candles
        # so 1m SuperTrend is also ready.
        interval = int(config.get('candle_interval', 5) or 5)
        if bool(config.get('htf_filter_enabled', True)) and interval < 60 and (60 % max(1, interval) == 0):
            multiple_1m = 60 // max(1, interval)
            base_needed = max(base_needed, multiple_1m * (st_period + 1))

        # If score engine is selected, ensure both base TF and next TF are ready.
        if str(config.get('indicator_type', 'supertrend') or 'supertrend').strip().lower() == 'score_mds':
            try:
                base_tf = int(config.get('candle_interval', 5) or 5)
                if base_tf in ScoreEngine._TF_CHAIN:
                    chain = list(ScoreEngine._TF_CHAIN)
                    next_tf = chain[chain.index(base_tf) + 1]
                    multiple = int(next_tf) // max(1, base_tf)
                    next_tf_needed = max(st_period + 1, macd_slow + macd_signal)
                    base_needed = max(base_needed, multiple * next_tf_needed)
            except Exception:
                pass

        # Small safety cushion for flip/slope computations.
        return int(max(50, base_needed + 5))

    async def _seed_indicators_from_mds_history(self) -> None:
        if str(config.get('market_data_provider', 'dhan') or 'dhan').strip().lower() != 'mds':
            return
        if not bool(config.get('prefetch_candles_on_start', True)):
            return

        base_url = str(config.get('mds_base_url', '') or '').strip()
        if not base_url:
            return

        index_name = str(config.get('selected_index', 'NIFTY') or 'NIFTY').strip().upper()
        interval = int(config.get('candle_interval', 5) or 5)
        limit = self._prefetch_candles_needed()

        try:
            from mds_client import fetch_last_candles

            candles = await fetch_last_candles(
                base_url=base_url,
                symbol=index_name,
                timeframe_seconds=interval,
                limit=limit,
            )
        except Exception as e:
            logger.warning(f"[WARMUP] Prefetch failed (MDS): {e}")
            return

        if not candles:
            logger.info("[WARMUP] No candles returned from MDS (skipping seed)")
            return

        # Reset any MDS-derived HTF aggregation state
        self._mds_htf_count = 0
        self._mds_htf_high = 0.0
        self._mds_htf_low = float('inf')
        self._mds_htf_close = 0.0

        multiple_1m = None
        if bool(config.get('htf_filter_enabled', True)) and interval < 60 and (60 % max(1, interval) == 0):
            multiple_1m = 60 // max(1, interval)

        last_indicator_value = None
        last_signal = None
        last_mds = None

        for row in candles:
            if not isinstance(row, dict):
                continue
            try:
                high = float(row.get('high') or 0.0)
                low = float(row.get('low') or float('inf'))
                close = float(row.get('close') or 0.0)
            except Exception:
                continue

            if close <= 0 or high <= 0 or low == float('inf'):
                continue

            last_indicator_value, last_signal = self.indicator.add_candle(high, low, close)
            if self.macd:
                self.macd.add_candle(high, low, close)

            if str(config.get('indicator_type') or '').strip().lower() == 'score_mds' and self.score_engine:
                try:
                    last_mds = self.score_engine.on_base_candle(Candle(high=high, low=low, close=close))
                except Exception:
                    last_mds = None

            if multiple_1m:
                self._mds_htf_count += 1
                self._mds_htf_high = max(self._mds_htf_high, high)
                self._mds_htf_low = min(self._mds_htf_low, low)
                self._mds_htf_close = close

                if self._mds_htf_count >= multiple_1m:
                    htf_value, htf_signal = self.htf_indicator.add_candle(
                        self._mds_htf_high, self._mds_htf_low, self._mds_htf_close
                    )
                    if htf_value:
                        bot_state['htf_supertrend_value'] = htf_value if isinstance(htf_value, (int, float)) else str(htf_value)
                        if htf_signal == 'GREEN':
                            bot_state['htf_signal_status'] = 'buy'
                        elif htf_signal == 'RED':
                            bot_state['htf_signal_status'] = 'sell'
                        else:
                            bot_state['htf_signal_status'] = 'waiting'
                        if htf_signal:
                            bot_state['htf_supertrend_signal'] = htf_signal

                    self._mds_htf_count = 0
                    self._mds_htf_high = 0.0
                    self._mds_htf_low = float('inf')
                    self._mds_htf_close = 0.0

        # Publish last computed values to state so UI doesn't show "waiting" on startup.
        if last_indicator_value:
            bot_state['supertrend_value'] = last_indicator_value if isinstance(last_indicator_value, (int, float)) else str(last_indicator_value)
            if last_signal == 'GREEN':
                bot_state['signal_status'] = 'buy'
            elif last_signal == 'RED':
                bot_state['signal_status'] = 'sell'
            else:
                bot_state['signal_status'] = 'waiting'
            if last_signal:
                bot_state['last_supertrend_signal'] = last_signal

        if self.macd and self.macd.last_macd is not None:
            bot_state['macd_value'] = float(self.macd.last_macd)

        if last_mds is not None:
            bot_state['mds_score'] = float(last_mds.score)
            bot_state['mds_slope'] = float(last_mds.slope)
            bot_state['mds_acceleration'] = float(last_mds.acceleration)
            bot_state['mds_stability'] = float(last_mds.stability)
            bot_state['mds_confidence'] = float(last_mds.confidence)
            bot_state['mds_is_choppy'] = bool(last_mds.is_choppy)
            bot_state['mds_direction'] = str(last_mds.direction)

        # Prevent immediate re-processing of the last candle on first poll.
        try:
            self._last_mds_candle_ts = str((candles[-1] or {}).get('ts') or '') or None
        except Exception:
            self._last_mds_candle_ts = None

        logger.info(f"[WARMUP] Seeded indicators from MDS history | Candles={len(candles)} Interval={interval}s")

    async def _handle_closed_candle(
        self,
        *,
        index_name: str,
        candle_number: int,
        candle_interval: int,
        candle_open: float,
        high: float,
        low: float,
        close: float,
        current_candle_time: datetime,
    ) -> None:
        if not (high > 0 and low < float('inf') and close > 0):
            return

        indicator_value, signal = self.indicator.add_candle(high, low, close)

        # MACD: keep both line and histogram; entry spec uses histogram.
        macd_line_value: Optional[float] = None
        macd_hist_value: Optional[float] = None
        if self.macd:
            _macd_line, _macd_cross = self.macd.add_candle(high, low, close)
            if self.macd.last_macd is not None:
                macd_line_value = float(self.macd.last_macd)
            if self.macd.last_histogram is not None:
                macd_hist_value = float(self.macd.last_histogram)

        mds_snapshot = None
        if config.get('indicator_type') == 'score_mds' and self.score_engine:
            try:
                mds_snapshot = self.score_engine.on_base_candle(Candle(high=float(high), low=float(low), close=float(close)))

                bot_state['mds_score'] = float(mds_snapshot.score)
                bot_state['mds_slope'] = float(mds_snapshot.slope)
                bot_state['mds_acceleration'] = float(mds_snapshot.acceleration)
                bot_state['mds_stability'] = float(mds_snapshot.stability)
                bot_state['mds_confidence'] = float(mds_snapshot.confidence)
                bot_state['mds_is_choppy'] = bool(mds_snapshot.is_choppy)
                bot_state['mds_direction'] = str(mds_snapshot.direction)
            except Exception as e:
                logger.error(f"[MDS] ScoreEngine update failed: {e}", exc_info=True)
                mds_snapshot = None

        # Update state if indicator is ready
        if indicator_value:
            bot_state['supertrend_value'] = indicator_value if isinstance(indicator_value, (int, float)) else str(indicator_value)

        # ADX update (optional; used by regime detector when enabled)
        if self.adx:
            try:
                adx_val, _adx_sig = self.adx.add_candle(high, low, close)
                bot_state['adx_value'] = None if adx_val is None else float(adx_val)
            except Exception:
                bot_state['adx_value'] = None
        if macd_line_value is not None:
            bot_state['macd_value'] = float(macd_line_value)
        else:
            bot_state['macd_value'] = 0.0

        # Expose histogram as telemetry (used by v2 logic)
        bot_state['macd_histogram'] = 0.0 if macd_hist_value is None else float(macd_hist_value)

        # Update signal status (GREEN="buy", RED="sell", None="waiting")
        if signal == "GREEN":
            bot_state['signal_status'] = "buy"
        elif signal == "RED":
            bot_state['signal_status'] = "sell"
        else:
            bot_state['signal_status'] = "waiting"

        # Always log candle close (even while indicators warm up)
        st_txt = f"{indicator_value:.2f}" if isinstance(indicator_value, (int, float)) else "NA"
        macd_txt = f"{(macd_line_value or 0.0):.4f}" if isinstance(macd_line_value, (int, float)) else "NA"
        signal_txt = signal if signal else "NONE"
        logger.info(
            f"[CANDLE CLOSE #{candle_number}] {index_name} | "
            f"H={high:.2f} L={low:.2f} C={close:.2f} | "
            f"ST={st_txt} | MACD={macd_txt} | Signal={signal_txt}"
        )

        # Save candle data for analysis (optional; disabled by default to keep DB small)
        if indicator_value and config.get('store_candle_data', False):
            from database import save_candle_data

            await save_candle_data(
                candle_number=candle_number,
                index_name=index_name,
                high=high,
                low=low,
                close=close,
                supertrend_value=indicator_value,
                macd_value=macd_value,
                signal_status=bot_state['signal_status'],
                interval_seconds=int(config.get('candle_interval', candle_interval) or candle_interval),
            )

        # Candle-close trailing SL/target check regardless of signal readiness
        if self.current_position:
            option_ltp = bot_state['current_option_ltp']
            sl_hit = await self.check_trailing_sl_on_close(option_ltp)
            if sl_hit:
                self.last_exit_candle_time = current_candle_time

        # ═══════════════════════════════════════════════════════════════════
        # NEW: Trading Logic v2 - Market Regime, Confidence, Entry/Exit
        # ═══════════════════════════════════════════════════════════════════
        
        # v2 logic is always-on for SuperTrend-based modes.
        # In score_mds mode, ScoreEngine has its own trade decisions; avoid double-trading.
        use_v2_logic = str(config.get('indicator_type') or '').strip().lower() != 'score_mds'

        if indicator_value and isinstance(indicator_value, (int, float)) and use_v2_logic:
            current_price = float(bot_state.get('index_ltp') or 0.0)

            # Track candle range history for volatility expansion
            candle_range = float(high) - float(low)
            if candle_range > 0:
                self._recent_candle_ranges.append(candle_range)
                if len(self._recent_candle_ranges) > 200:
                    self._recent_candle_ranges = self._recent_candle_ranges[-200:]

            # Market regime detection (gatekeeper)
            regime, regime_info = self.regime_detector.detect(
                current_price=current_price,
                supertrend_value=float(indicator_value),
                adx=float(bot_state.get('adx_value')) if bot_state.get('adx_value') is not None else None,
            )
            bot_state['market_regime'] = regime.value

            # Cooldown advances on every candle
            self.cooldown_mgr.advance()
            cd_state = self.cooldown_mgr.get_state()
            bot_state['cooldown_active'] = bool(cd_state.active)
            bot_state['cooldown_candles_remaining'] = int(cd_state.candles_remaining)

            # Candle body% (spec)
            body_pct = 0.0
            if float(high) != float(low):
                body_pct = abs(float(close) - float(candle_open)) / (float(high) - float(low)) * 100.0

            # Runtime: update confidence & evaluate exits if position open
            if self.current_position:
                self.candles_in_position += 1

                new_conf, conf_reason = self.confidence_calc.calculate_runtime_confidence(
                    previous_confidence=float(self.runtime_confidence),
                    candle_body_pct=float(body_pct),
                    macd_histogram=macd_hist_value,
                    prev_macd_histogram=self.prev_macd_histogram,
                    candles_held=int(self.candles_in_position),
                )
                self.runtime_confidence = float(new_conf)
                bot_state['runtime_confidence'] = float(self.runtime_confidence)

                # Risk-first exits (spec order)
                option_ltp = float(bot_state.get('current_option_ltp') or 0.0)
                position_type = str(self.current_position.get('option_type') or '').upper()
                qty = int(self.current_position.get('qty') or 0)
                if qty <= 0:
                    index_config = get_index_config(index_name)
                    qty = int(config.get('order_qty', 1) or 1) * int(index_config['lot_size'])

                initial_sl = float(config.get('initial_stoploss', 0) or 0.0)
                hard_sl = (float(self.entry_price) - initial_sl) if initial_sl > 0 else None

                exit_signal = self.exit_logic.check_exit(
                    is_open_position=True,
                    current_option_ltp=option_ltp,
                    hard_sl=hard_sl,
                    trailing_sl=self.trailing_sl,
                    runtime_confidence=float(self.runtime_confidence),
                    min_runtime_confidence=float(config.get('min_runtime_confidence', 40.0) or 40.0),
                    st_direction=int(self.indicator.direction) if self.indicator else None,
                    prev_st_direction=int(self.prev_st_direction) if self.prev_st_direction is not None else None,
                    position_type=position_type,
                    current_time_ist=get_ist_time().time(),
                    bypass_eod_exit=bool(config.get('bypass_market_hours', False)),
                )

                if exit_signal.should_exit and exit_signal.reason is not None:
                    pnl = (option_ltp - float(self.entry_price)) * float(qty)
                    logger.warning(f"[V2_EXIT] {exit_signal.reason.value} | Conf={self.runtime_confidence:.0f} ({conf_reason}) | BodyPct={body_pct:.0f} | PnL={pnl:.2f}")
                    closed = await self.close_position(option_ltp, pnl, exit_signal.reason.value)
                    if closed:
                        self.last_exit_candle_time = current_candle_time

            # Entry: only if no open position AND not in cooldown AND trending regime
            else:
                self.candles_in_position = 0

                if regime == MarketRegime.SIDEWAYS:
                    logger.debug("[V2_ENTRY] Skipped (SIDEWAYS regime)")
                elif not self.cooldown_mgr.can_enter():
                    logger.debug("[V2_ENTRY] Skipped (cooldown active)")
                elif not self.is_within_trading_hours() or bool(bot_state.get('daily_max_loss_triggered', False)):
                    logger.debug("[V2_ENTRY] Skipped (hours/maxloss gate)")
                else:
                    st_distance_pct = (abs(current_price - float(indicator_value)) / current_price * 100.0) if current_price > 0 else 0.0

                    confidence_bd = self.confidence_calc.calculate_entry_confidence(
                        supertrend_distance_pct=float(st_distance_pct),
                        candle_open=float(candle_open),
                        candle_high=float(high),
                        candle_low=float(low),
                        candle_close=float(close),
                        macd_histogram=macd_hist_value,
                        recent_ranges=list(self._recent_candle_ranges),
                    )
                    bot_state['confidence_entry'] = float(confidence_bd.total)
                    logger.info(f"[V2_ENTRY_CONF] {confidence_bd}")

                    entry_decision = self.entry_logic.check_entry(
                        market_regime_is_trending=True,
                        st_direction=int(self.indicator.direction),
                        prev_st_direction=int(self.prev_st_direction) if self.prev_st_direction is not None else None,
                        supertrend_value=float(indicator_value),
                        candle_close=float(close),
                        macd_histogram=macd_hist_value,
                        entry_confidence=float(confidence_bd.total),
                        min_entry_confidence=float(config.get('min_entry_confidence', 65.0) or 65.0),
                    )

                    if entry_decision.should_enter and entry_decision.direction is not None:
                        option_type = 'CE' if entry_decision.direction == EntryDirection.BULLISH else 'PE'
                        atm_strike = round_to_strike(current_price, index_name)
                        logger.info(f"[V2_ENTRY] YES | {option_type} | ATM={atm_strike} | Conf={confidence_bd.total:.0f}")
                        await self.enter_position(option_type, atm_strike, current_price)
                        self.entry_confidence_recorded = float(confidence_bd.total)
                        self.runtime_confidence = 100.0
                        self.candles_in_position = 0
                    else:
                        logger.debug(f"[V2_ENTRY] NO | Reason={entry_decision.reason} | Conf={confidence_bd.total:.0f}")

            # Track previous MACD histogram and ST direction
            self.prev_macd_histogram = macd_hist_value
            self.prev_st_direction = int(self.indicator.direction)

        # ═══════════════════════════════════════════════════════════════════
        # END: Trading Logic v2
        # ═══════════════════════════════════════════════════════════════════

        # Score-engine trading path (independent of SuperTrend signal emission)
        if config.get('indicator_type') == 'score_mds' and mds_snapshot is not None:
            can_trade = True
            if self.last_exit_candle_time:
                time_since_exit = (current_candle_time - self.last_exit_candle_time).total_seconds()
                if time_since_exit < candle_interval:
                    can_trade = False
            if can_trade:
                exited = await self.process_mds_on_close(mds_snapshot, close)
                if exited:
                    self.last_exit_candle_time = current_candle_time
        else:
            # Prevent double-trading: v2 is authoritative for SuperTrend-based modes.
            if str(config.get('indicator_type') or '').strip().lower() != 'score_mds':
                return

            prev_signal = bot_state.get('last_supertrend_signal')
            flipped = bool(signal) and (prev_signal is None or signal != prev_signal)
            if signal:
                bot_state['last_supertrend_signal'] = signal

            # Trading logic - entries/exits based on SuperTrend signal
            can_trade = True
            if self.last_exit_candle_time:
                time_since_exit = (current_candle_time - self.last_exit_candle_time).total_seconds()
                if time_since_exit < candle_interval:
                    can_trade = False

            if not can_trade:
                logger.info("[ENTRY_DECISION] NO | Reason=recent_exit_cooldown")
            else:
                exited = await self.process_signal_on_close(str(signal or ''), close, flipped=bool(flipped))
                if exited:
                    self.last_exit_candle_time = current_candle_time

    def _can_place_new_entry_order(self) -> bool:
        cooldown = int(config.get('min_order_cooldown_seconds', 0) or 0)
        if cooldown <= 0 or self.last_order_time_utc is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self.last_order_time_utc).total_seconds()
        return elapsed >= cooldown

    def _remaining_entry_cooldown(self) -> float:
        cooldown = int(config.get('min_order_cooldown_seconds', 0) or 0)
        if cooldown <= 0 or self.last_order_time_utc is None:
            return 0.0
        elapsed = (datetime.now(timezone.utc) - self.last_order_time_utc).total_seconds()
        return max(0.0, cooldown - elapsed)

    def _min_hold_active(self) -> bool:
        min_hold = int(config.get('min_hold_seconds', 0) or 0)
        if min_hold <= 0 or self.entry_time_utc is None or not self.current_position:
            return False
        held = (datetime.now(timezone.utc) - self.entry_time_utc).total_seconds()
        return held < min_hold
    
    def initialize_dhan(self):
        """Initialize Dhan API connection"""
        if config['dhan_access_token'] and config['dhan_client_id']:
            try:
                self.dhan = DhanAPI(config['dhan_access_token'], config['dhan_client_id'])
                logger.info("[MARKET] Dhan API initialized")
                return True
            except Exception as e:
                self.dhan = None
                logger.warning(f"[ERROR] Dhan API init failed: {e}")
                return False
        logger.warning("[ERROR] Dhan API credentials not configured")
        return False

    async def _init_paper_replay(self) -> None:
        """Load candle data from DB for after-hours paper replay."""
        try:
            index_name = config.get('selected_index', 'NIFTY')
            interval = int(config.get('candle_interval', 5) or 5)
            date_ist = str(config.get('paper_replay_date_ist', '') or '').strip() or None

            provider = str(config.get('market_data_provider', 'dhan') or 'dhan').strip().lower()
            base_url = str(config.get('mds_base_url', '') or '').strip()

            candles = []
            if provider == 'mds' and base_url and date_ist:
                # Consume-only replay: pull historical candles for the selected IST date
                # from market-data-service (TimescaleDB) and replay them locally.
                from mds_client import fetch_candles_for_ist_date

                candles = await fetch_candles_for_ist_date(
                    base_url=base_url,
                    symbol=index_name,
                    timeframe_seconds=interval,
                    date_ist=date_ist,
                    limit=200000,
                )
            else:
                # Legacy fallback: backend SQLite candle_data table (may be empty in consume-only setups)
                from database import get_candle_data_for_replay

                candles = await get_candle_data_for_replay(
                    index_name=index_name,
                    interval_seconds=interval,
                    date_ist=date_ist,
                    limit=20000,
                )

            self._paper_replay_candles = candles or []
            self._paper_replay_pos = 0
            self._paper_replay_htf_elapsed = 0

            src = "MDS" if (provider == 'mds' and base_url and date_ist) else "SQLITE"
            logger.info(f"[REPLAY] Loaded {len(self._paper_replay_candles)} candles | Source={src} | Index={index_name} Interval={interval}s DateIST={date_ist or 'latest'}")
        except Exception as e:
            self._paper_replay_candles = []
            self._paper_replay_pos = 0
            self._paper_replay_htf_elapsed = 0
            logger.error(f"[REPLAY] Failed to load candles: {e}")
    
    def _initialize_indicator(self):
        """Initialize indicators (SuperTrend + optional MACD confirmation)"""
        try:
            self.indicator = SuperTrend(
                period=config['supertrend_period'],
                multiplier=config['supertrend_multiplier']
            )
            # HTF indicator uses same parameters; timeframe aggregation handled in run_loop
            self.htf_indicator = SuperTrend(
                period=config['supertrend_period'],
                multiplier=config['supertrend_multiplier']
            )
            # MACD used for entry confirmation (if enabled)
            self.macd = MACD(
                fast=int(config.get('macd_fast', 12)),
                slow=int(config.get('macd_slow', 26)),
                signal=int(config.get('macd_signal', 9)),
            )

            # Optional: ADX for market regime detection
            self.adx = ADX(period=14) if bool(config.get('use_adx_for_regime', False)) else None

            self.score_engine = ScoreEngine(
                st_period=int(config.get('supertrend_period', 7)),
                st_multiplier=float(config.get('supertrend_multiplier', 4)),
                macd_fast=int(config.get('macd_fast', 12)),
                macd_slow=int(config.get('macd_slow', 26)),
                macd_signal=int(config.get('macd_signal', 9)),
                base_timeframe_seconds=int(config.get('candle_interval', 5) or 5),
                bonus_macd_triple=float(config.get('mds_bonus_macd_triple', 1.0) or 0.0),
                bonus_macd_momentum=float(config.get('mds_bonus_macd_momentum', 0.5) or 0.0),
                bonus_macd_cross=float(config.get('mds_bonus_macd_cross', 0.5) or 0.0),
            )
            logger.info(f"[SIGNAL] SuperTrend initialized")
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize indicator: {e}")
            # Fallback to SuperTrend
            self.indicator = SuperTrend(period=7, multiplier=4)
            self.htf_indicator = SuperTrend(period=7, multiplier=4)
            self.macd = MACD(fast=12, slow=26, signal=9)
            self.adx = ADX(period=14) if bool(config.get('use_adx_for_regime', False)) else None
            self.score_engine = ScoreEngine(
                st_period=7,
                st_multiplier=4,
                macd_fast=12,
                macd_slow=26,
                macd_signal=9,
                base_timeframe_seconds=int(config.get('candle_interval', 5) or 5),
                bonus_macd_triple=float(config.get('mds_bonus_macd_triple', 1.0) or 0.0),
                bonus_macd_momentum=float(config.get('mds_bonus_macd_momentum', 0.5) or 0.0),
                bonus_macd_cross=float(config.get('mds_bonus_macd_cross', 0.5) or 0.0),
            )
            logger.info(f"[SIGNAL] SuperTrend (fallback) initialized")
    
    def reset_indicator(self):
        """Reset the selected indicator"""
        if self.indicator:
            self.indicator.reset()
        if self.htf_indicator:
            self.htf_indicator.reset()
        if self.macd:
            self.macd.reset()
        if self.adx:
            self.adx.reset()
        if self.score_engine:
            self.score_engine.reset()
        else:
            # If score_engine wasn't initialized for some reason, try to initialize now.
            try:
                self.score_engine = ScoreEngine(
                    st_period=int(config.get('supertrend_period', 7)),
                    st_multiplier=float(config.get('supertrend_multiplier', 4)),
                    macd_fast=int(config.get('macd_fast', 12)),
                    macd_slow=int(config.get('macd_slow', 26)),
                    macd_signal=int(config.get('macd_signal', 9)),
                    base_timeframe_seconds=int(config.get('candle_interval', 5) or 5),
                    bonus_macd_triple=float(config.get('mds_bonus_macd_triple', 1.0) or 0.0),
                    bonus_macd_momentum=float(config.get('mds_bonus_macd_momentum', 0.5) or 0.0),
                    bonus_macd_cross=float(config.get('mds_bonus_macd_cross', 0.5) or 0.0),
                )
            except Exception:
                self.score_engine = None

        self._mds_last_direction = None
        self._mds_confirm_count = 0
        self._last_mds_candle_ts = None
        self._mds_htf_count = 0
        self._mds_htf_high = 0.0
        self._mds_htf_low = float('inf')
        self._mds_htf_close = 0.0
        
        # NEW: Reset Trading Logic v2 State
        self.runtime_confidence = 100.0
        self.entry_confidence_recorded = 0.0
        self.prev_st_direction = None
        self.candles_in_position = 0
        self.prev_macd_histogram = None
        self.cooldown_mgr.reset()
        
        logger.info(f"[SIGNAL] Indicator reset: {config.get('indicator_type', 'supertrend')}")
        logger.info("[SIGNAL] Trading Logic v2 state reset")
    
    def is_within_trading_hours(self) -> bool:
        """Check if current time allows new entries
        
        Returns:
            bool: True if within allowed trading hours, False otherwise
        """
        if config.get('bypass_market_hours', False) or (
            bot_state.get('mode') == 'paper' and config.get('paper_replay_enabled', False)
        ):
            return True

        ist = get_ist_time()
        current_time = ist.time()
        
        # Define trading hours
        NO_ENTRY_BEFORE = datetime.strptime("09:25", "%H:%M").time()  # No entry before 9:25 AM
        NO_ENTRY_AFTER = datetime.strptime("15:10", "%H:%M").time()   # No entry after 3:10 PM
        
        # Check if within allowed hours
        if current_time < NO_ENTRY_BEFORE:
            logger.info(f"[HOURS] Entry blocked - market not open yet (Current: {current_time.strftime('%H:%M')}, Opens: 09:25)")
            return False
        
        if current_time > NO_ENTRY_AFTER:
            logger.info(f"[HOURS] Entry blocked - market closing soon (Current: {current_time.strftime('%H:%M')}, Cutoff: 15:10)")
            return False
        
        logger.debug(f"[HOURS] Trading hours OK (Current: {current_time.strftime('%H:%M')})")
        return True
    
    async def start(self):
        """Start the trading bot"""
        if self.running:
            return {"status": "error", "message": "Bot already running"}
        
        # Only require Dhan in live mode. Paper mode can run without broker SDK.
        if bot_state.get('mode') != 'paper':
            if not self.initialize_dhan():
                return {"status": "error", "message": "Dhan API not available (credentials/SDK)"}

        # Prepare replay candles for after-hours (or bypass-hours) paper simulation
        replay_enabled = bool(bot_state.get('mode') == 'paper' and config.get('paper_replay_enabled', False))
        if replay_enabled:
            await self._init_paper_replay()
            if not self._paper_replay_candles:
                return {"status": "error", "message": "Paper replay enabled but no candles found (MDS/DB)"}
        
        self.running = True
        bot_state['is_running'] = True
        self.reset_indicator()
        self.last_signal = None

        # Do not seed from "latest" history when running a dated replay.
        # Also skip MDS prefetch when running synthetic-only paper testing.
        synthetic_only = (
            bot_state.get('mode') == 'paper'
            and bool(config.get('bypass_market_hours', False))
            and not bool(config.get('paper_replay_enabled', False))
        )
        if (not replay_enabled) and (not synthetic_only):
            # Prefetch and seed indicators from Timescale via market-data-service so we don't
            # spend the first N candles "warming up" after each restart.
            await self._seed_indicators_from_mds_history()

        self.task = asyncio.create_task(self.run_loop())
        
        index_name = config['selected_index']
        interval = format_timeframe(config['candle_interval'])
        indicator_name = config.get('indicator_type', 'supertrend')
        logger.info(f"[BOT] Started - Index: {index_name}, Timeframe: {interval}, Indicator: {indicator_name}, Mode: {bot_state['mode']}")
        
        return {"status": "success", "message": f"Bot started for {index_name} ({interval})"}
    
    async def stop(self):
        """Stop the trading bot"""
        self.running = False
        bot_state['is_running'] = False
        if self.task:
            self.task.cancel()
        logger.info("[BOT] Stopped")
        return {"status": "success", "message": "Bot stopped"}
    
    async def squareoff(self):
        """Force square off current position"""
        if not self.current_position:
            return {"status": "error", "message": "No open position"}
        
        index_name = config['selected_index']
        qty = int(self.current_position.get('qty') or 0)
        if qty <= 0:
            index_config = get_index_config(index_name)
            qty = config['order_qty'] * index_config['lot_size']
        
        logger.info(f"[ORDER] Force squareoff initiated for {index_name}")
        
        exit_price = bot_state['current_option_ltp']
        pnl = (exit_price - self.entry_price) * qty
        closed = await self.close_position(exit_price, pnl, "Force Square-off")
        if closed:
            suffix = "(Paper)" if bot_state['mode'] == 'paper' else ""
            return {"status": "success", "message": f"Position squared off {suffix}. PnL: {pnl:.2f}"}
        return {"status": "error", "message": "Squareoff order not filled (position still open)"}
    
    async def close_position(self, exit_price: float, pnl: float, reason: str) -> bool:
        """Close current position and save trade"""
        if not self.current_position:
            return False
        
        trade_id = self.current_position.get('trade_id', '')
        index_name = self.current_position.get('index_name', config['selected_index'])
        option_type = self.current_position.get('option_type', '')
        strike = self.current_position.get('strike', 0)
        security_id = self.current_position.get('security_id', '')
        qty = int(self.current_position.get('qty') or 0)
        if qty <= 0:
            index_config = get_index_config(index_name)
            qty = config['order_qty'] * index_config['lot_size']
        
        exit_order_placed = False
        filled_exit_price = exit_price

        if bot_state['mode'] != 'paper' and self.dhan and security_id:
            existing_exit_order_id = self.current_position.get('exit_order_id')

            try:
                if not existing_exit_order_id:
                    logger.info(f"[ORDER] Placing EXIT SELL order | Trade ID: {trade_id} | Security: {security_id} | Qty: {qty}")
                    result = await self.dhan.place_order(security_id, "SELL", qty, index_name=index_name)

                    if result.get('status') == 'success' and result.get('orderId'):
                        existing_exit_order_id = result.get('orderId')
                        self.current_position['exit_order_id'] = existing_exit_order_id
                        bot_state['current_position'] = self.current_position
                        exit_order_placed = True
                        self.last_order_time_utc = datetime.now(timezone.utc)
                        logger.info(f"[ORDER] ✓ EXIT order PLACED | OrderID: {existing_exit_order_id} | Security: {security_id} | Qty: {qty}")
                    else:
                        logger.error(f"[ORDER] ✗ EXIT order FAILED | Trade: {trade_id} | Result: {result}")
                        return False

                verify = await self.dhan.verify_order_filled(
                    order_id=str(existing_exit_order_id),
                    security_id=str(security_id),
                    expected_qty=int(qty),
                    timeout_seconds=30
                )

                if not verify.get('filled'):
                    status = verify.get('status')
                    logger.warning(
                        f"[ORDER] ✗ EXIT not filled yet | Trade: {trade_id} | OrderID: {existing_exit_order_id} | Status: {status} | {verify.get('message')}"
                    )
                    if status in {"REJECTED", "CANCELLED", "ERROR"}:
                        self.current_position.pop('exit_order_id', None)
                        bot_state['current_position'] = self.current_position
                    return False

                avg_price = float(verify.get('average_price') or 0)
                if avg_price > 0:
                    filled_exit_price = round(avg_price / 0.05) * 0.05
                    filled_exit_price = round(filled_exit_price, 2)
                exit_order_placed = True

            except Exception as e:
                logger.error(f"[ORDER] ✗ Error placing/verifying EXIT order: {e} | Trade: {trade_id}", exc_info=True)
                return False
        elif not security_id:
            logger.warning(f"[WARNING] Cannot send exit order - security_id missing for {index_name} {option_type} | Trade: {trade_id}")
            logger.info(f"[EXIT] ✓ Position closed | {index_name} {option_type} {strike} | Reason: {reason} | PnL: {pnl} | Order Placed: False")
            # Update DB in background - don't wait
            asyncio.create_task(update_trade_exit(
                trade_id=trade_id,
                exit_time=datetime.now(timezone.utc).isoformat(),
                exit_price=exit_price,
                pnl=pnl,
                exit_reason=reason
            ))
            # Track last order timestamp (treated as an exit action even if no broker order)
            self.last_order_time_utc = datetime.now(timezone.utc)
        elif bot_state['mode'] == 'paper':
            logger.info(f"[ORDER] Paper mode - EXIT order not placed to Dhan (simulated) | Trade: {trade_id}")
            logger.info(f"[EXIT] ✓ Position closed | {index_name} {option_type} {strike} | Reason: {reason} | PnL: {pnl} | Order Placed: False")
            # Update DB in background - don't wait
            asyncio.create_task(update_trade_exit(
                trade_id=trade_id,
                exit_time=datetime.now(timezone.utc).isoformat(),
                exit_price=exit_price,
                pnl=pnl,
                exit_reason=reason
            ))
            # Track last order timestamp (paper exit)
            self.last_order_time_utc = datetime.now(timezone.utc)

        # If we reached here in LIVE mode, the exit is filled. Use filled price for P&L if available.
        if bot_state['mode'] != 'paper' and self.dhan and security_id:
            pnl = (filled_exit_price - self.entry_price) * qty

            asyncio.create_task(update_trade_exit(
                trade_id=trade_id,
                exit_time=datetime.now(timezone.utc).isoformat(),
                exit_price=filled_exit_price,
                pnl=pnl,
                exit_reason=reason
            ))
        
        # Update state
        bot_state['daily_pnl'] += pnl
        bot_state['current_position'] = None
        bot_state['trailing_sl'] = None
        bot_state['entry_price'] = 0
        
        if bot_state['daily_pnl'] < -config['daily_max_loss']:
            bot_state['daily_max_loss_triggered'] = True
            logger.warning(f"[EXIT] Daily max loss triggered! PnL: {bot_state['daily_pnl']:.2f}")
        
        if pnl < 0 and abs(pnl) > bot_state['max_drawdown']:
            bot_state['max_drawdown'] = abs(pnl)
        
        # Track the signal at exit - require signal change before next entry
        # If we exited CE position, last signal was GREEN
        # If we exited PE position, last signal was RED
        if option_type == 'CE':
            self.last_signal = 'GREEN'
        elif option_type == 'PE':
            self.last_signal = 'RED'
        
        self.current_position = None
        self.entry_price = 0
        self.trailing_sl = None
        self.highest_profit = 0
        self.entry_time_utc = None

        # NEW: v2 cooldown activates after ANY exit (spec: mandatory cooldown)
        try:
            self.cooldown_mgr.activate(reason=CooldownReason.EXIT)
            state = self.cooldown_mgr.get_state()
            bot_state['cooldown_active'] = bool(state.active)
            bot_state['cooldown_candles_remaining'] = int(state.candles_remaining)
        except Exception:
            pass

        logger.info(f"[EXIT] ✓ Position closed | {index_name} {option_type} {strike} | Reason: {reason} | PnL: {pnl:.2f} | Order Placed: {exit_order_placed}")
        return True
    
    async def run_loop(self):
        """Main trading loop"""
        logger.info("[BOT] Trading loop started")
        candle_start_time = datetime.now()
        candle_open = 0.0
        high, low, close = 0, float('inf'), 0
        candle_number = 0

        # Higher timeframe (HTF) candle aggregation (default: 1 minute)
        htf_candle_start_time = datetime.now()
        htf_high, htf_low, htf_close = 0, float('inf'), 0
        htf_candle_number = 0
        
        while self.running:
            try:
                index_name = config['selected_index']
                candle_interval = config['candle_interval']

                replay_enabled = (
                    bot_state.get('mode') == 'paper' and bool(config.get('paper_replay_enabled', False))
                )
                
                # Check daily reset (9:15 AM IST)
                ist = get_ist_time()
                if ist.hour == 9 and ist.minute == 15:
                    bot_state['daily_trades'] = 0
                    bot_state['daily_pnl'] = 0.0
                    bot_state['daily_max_loss_triggered'] = False
                    bot_state['max_drawdown'] = 0.0
                    self.last_exit_candle_time = None
                    self.last_trade_time = None
                    self.last_signal = None
                    candle_number = 0
                    htf_candle_number = 0
                    self.reset_indicator()
                    logger.info("[BOT] Daily reset at 9:15 AM")
                
                # Force square-off at 3:25 PM (skip during paper replay)
                if (not replay_enabled) and should_force_squareoff() and self.current_position:
                    logger.info("[EXIT] Auto squareoff at 3:25 PM")
                    await self.squareoff()

                # Check if trading is allowed (skip during paper replay)
                if (not replay_enabled) and (not is_market_open()):
                    await asyncio.sleep(5)
                    continue
                
                if bot_state['daily_max_loss_triggered'] and not self.current_position:
                    await asyncio.sleep(5)
                    continue
                
                # Paper replay: drive candles from DB (after-hours simulation)
                if replay_enabled:
                    if self._paper_replay_pos >= len(self._paper_replay_candles):
                        logger.info("[REPLAY] Completed candle replay")
                        self.running = False
                        bot_state['is_running'] = False
                        break

                    row = self._paper_replay_candles[self._paper_replay_pos]
                    self._paper_replay_pos += 1

                    try:
                        candle_open = float(row.get('open') or row.get('close') or 0.0)
                        high = float(row.get('high') or 0.0)
                        low = float(row.get('low') or float('inf'))
                        close = float(row.get('close') or 0.0)
                    except Exception:
                        candle_open, high, low, close = 0.0, 0.0, float('inf'), 0.0

                    # Treat candle close price as current LTP
                    if close > 0:
                        bot_state['index_ltp'] = close

                    # Update HTF (replay-time based, not wall-clock)
                    if config.get('htf_filter_enabled', True) and int(candle_interval) < 60 and close > 0:
                        if close > htf_high:
                            htf_high = close
                        if close < htf_low:
                            htf_low = close
                        htf_close = close

                        htf_seconds = int(config.get('htf_filter_timeframe', 60) or 60)
                        if htf_seconds != 60:
                            htf_seconds = 60

                        self._paper_replay_htf_elapsed += int(candle_interval)
                        if self._paper_replay_htf_elapsed >= htf_seconds:
                            htf_candle_number += 1
                            if htf_high > 0 and htf_low < float('inf'):
                                htf_value, htf_signal = self.htf_indicator.add_candle(htf_high, htf_low, htf_close)
                                if htf_value:
                                    bot_state['htf_supertrend_value'] = htf_value if isinstance(htf_value, (int, float)) else str(htf_value)
                                    if htf_signal == "GREEN":
                                        bot_state['htf_signal_status'] = "buy"
                                    elif htf_signal == "RED":
                                        bot_state['htf_signal_status'] = "sell"
                                    else:
                                        bot_state['htf_signal_status'] = "waiting"
                                    if htf_signal:
                                        bot_state['htf_supertrend_signal'] = htf_signal

                            htf_high, htf_low, htf_close = 0, float('inf'), 0
                            self._paper_replay_htf_elapsed = 0

                    # Unified candle-close logic (v2 regime/entry/exit/cooldown)
                    candle_number += 1
                    if high > 0 and low < float('inf'):
                        current_candle_time = datetime.now()
                        await self._handle_closed_candle(
                            index_name=index_name,
                            candle_number=candle_number,
                            candle_interval=int(config.get('candle_interval', candle_interval) or candle_interval),
                            candle_open=float(candle_open or close or 0.0),
                            high=high,
                            low=low,
                            close=close,
                            current_candle_time=current_candle_time,
                        )

                    # Update simulated option pricing if needed
                    if self.current_position:
                        security_id = self.current_position.get('security_id', '')
                        if security_id.startswith('SIM_'):
                            strike = self.current_position.get('strike', 0)
                            option_type = self.current_position.get('option_type', '')
                            index_ltp = bot_state['index_ltp']
                            if strike and index_ltp:
                                distance_from_atm = abs(index_ltp - strike)
                                if option_type == 'CE':
                                    intrinsic = max(0, index_ltp - strike)
                                else:
                                    intrinsic = max(0, strike - index_ltp)
                                atm_time_value = 150
                                time_decay_factor = max(0, 1 - (distance_from_atm / 500))
                                time_value = atm_time_value * time_decay_factor
                                simulated_ltp = intrinsic + time_value
                                tick_movement = random.choice([-0.10, -0.05, 0, 0.05, 0.10])
                                simulated_ltp += tick_movement
                                simulated_ltp = round(simulated_ltp / 0.05) * 0.05
                                simulated_ltp = max(0.05, round(simulated_ltp, 2))
                                bot_state['current_option_ltp'] = simulated_ltp

                    await self.broadcast_state()
                    speed = float(config.get('paper_replay_speed', 10.0) or 10.0)
                    speed = max(0.1, min(100.0, speed))
                    await asyncio.sleep(max(0.05, float(candle_interval) / speed))
                    continue

                # Fetch market data (skip if MarketDataService is active)
                provider = str(config.get('market_data_provider', 'dhan') or 'dhan').strip().lower()

                # In paper mode, when bypass_market_hours is enabled and paper replay is OFF,
                # run fully synthetic candles even if docker env defaults provider to 'mds'.
                # This enables strategy testing outside market hours.
                if (
                    bot_state.get('mode') == 'paper'
                    and bool(config.get('bypass_market_hours', False))
                    and not replay_enabled
                ):
                    provider = 'synthetic'

                mds_new_candle = None
                mds_new_candle_ts = None

                # Prefer consuming candles from market-data-service (Timescale-backed)
                if provider == 'mds':
                    try:
                        from mds_client import fetch_latest_candle

                        base_url = str(config.get('mds_base_url', '') or '').strip()
                        poll_s = float(config.get('mds_poll_seconds', 1.0) or 1.0)
                        candle = await fetch_latest_candle(
                            base_url=base_url,
                            symbol=index_name,
                            timeframe_seconds=int(config.get('candle_interval', candle_interval) or candle_interval),
                            min_poll_seconds=poll_s,
                        )
                        if isinstance(candle, dict) and (candle.get('close') is not None):
                            try:
                                close_price = float(candle.get('close') or 0.0)
                            except Exception:
                                close_price = 0.0

                            if close_price > 0:
                                bot_state['index_ltp'] = float(close_price)

                                mds_new_candle_ts = str(candle.get('ts') or '') or None
                                if mds_new_candle_ts and mds_new_candle_ts != self._last_mds_candle_ts:
                                    mds_new_candle = candle
                                    self._last_mds_candle_ts = mds_new_candle_ts
                        elif self.dhan:
                            # If MDS has no candles yet, fall back to direct quote.
                            try:
                                index_ltp = await asyncio.to_thread(self.dhan.get_index_ltp, index_name)
                                if index_ltp and index_ltp > 0:
                                    bot_state['index_ltp'] = float(index_ltp)
                            except Exception:
                                pass
                    except Exception:
                        # If MDS is unavailable, fall back to any other provider.
                        pass

                    # Option LTP is only needed when a real position is open.
                    # MDS currently does not expose option quotes, so we use Dhan (if configured) in that case.
                    if self.dhan and self.current_position is not None:
                        security_id = self.current_position.get('security_id', '')
                        if security_id and not security_id.startswith('SIM_'):
                            try:
                                option_security_id = int(security_id)
                                _index_ltp, option_ltp = await asyncio.to_thread(
                                    self.dhan.get_index_and_option_ltp, index_name, option_security_id
                                )
                                if option_ltp and option_ltp > 0:
                                    option_ltp = round(option_ltp / 0.05) * 0.05
                                    bot_state['current_option_ltp'] = round(option_ltp, 2)
                            except Exception:
                                pass

                    # If we received a new candle from MDS, process it as the authoritative candle close.
                    if isinstance(mds_new_candle, dict):
                        try:
                            high = float(mds_new_candle.get('high') or 0.0)
                            low = float(mds_new_candle.get('low') or float('inf'))
                            close = float(mds_new_candle.get('close') or 0.0)
                        except Exception:
                            high, low, close = 0.0, float('inf'), 0.0

                        # Update HTF aggregation from base candles (fixed to 60s in current code).
                        interval = int(config.get('candle_interval', candle_interval) or candle_interval)
                        if bool(config.get('htf_filter_enabled', True)) and interval < 60 and (60 % max(1, interval) == 0) and close > 0:
                            multiple_1m = 60 // max(1, interval)
                            self._mds_htf_count += 1
                            self._mds_htf_high = max(self._mds_htf_high, high)
                            self._mds_htf_low = min(self._mds_htf_low, low)
                            self._mds_htf_close = close
                            if self._mds_htf_count >= multiple_1m:
                                htf_candle_number += 1
                                if self._mds_htf_high > 0 and self._mds_htf_low < float('inf'):
                                    htf_value, htf_signal = self.htf_indicator.add_candle(
                                        self._mds_htf_high, self._mds_htf_low, self._mds_htf_close
                                    )
                                    if htf_value:
                                        bot_state['htf_supertrend_value'] = htf_value if isinstance(htf_value, (int, float)) else str(htf_value)
                                        if htf_signal == 'GREEN':
                                            bot_state['htf_signal_status'] = 'buy'
                                        elif htf_signal == 'RED':
                                            bot_state['htf_signal_status'] = 'sell'
                                        else:
                                            bot_state['htf_signal_status'] = 'waiting'
                                        if htf_signal:
                                            bot_state['htf_supertrend_signal'] = htf_signal

                                self._mds_htf_count = 0
                                self._mds_htf_high = 0.0
                                self._mds_htf_low = float('inf')
                                self._mds_htf_close = 0.0

                        # Only advance candle number / trading decisions when the candle is valid.
                        if high > 0 and low < float('inf') and close > 0:
                            current_candle_time = datetime.now()
                            candle_number += 1
                            await self._handle_closed_candle(
                                index_name=index_name,
                                candle_number=candle_number,
                                candle_interval=int(config.get('candle_interval', candle_interval) or candle_interval),
                                candle_open=float(mds_new_candle.get('open') or close or 0.0) if isinstance(mds_new_candle, dict) else float(close or 0.0),
                                high=high,
                                low=low,
                                close=close,
                                current_candle_time=current_candle_time,
                            )

                # Legacy mode: fetch quotes directly via Dhan
                elif self.dhan and not bot_state.get('market_data_service_active', False):
                    has_position = self.current_position is not None
                    option_security_id = None
                    
                    if has_position:
                        security_id = self.current_position.get('security_id', '')
                        if security_id and not security_id.startswith('SIM_'):
                            option_security_id = int(security_id)
                    
                    # Fetch Index + Option LTP (Dhan SDK is sync; run in thread)
                    if option_security_id:
                        index_ltp, option_ltp = await asyncio.to_thread(
                            self.dhan.get_index_and_option_ltp, index_name, option_security_id
                        )
                        if index_ltp > 0:
                            bot_state['index_ltp'] = index_ltp
                        if option_ltp > 0:
                            option_ltp = round(option_ltp / 0.05) * 0.05
                            bot_state['current_option_ltp'] = round(option_ltp, 2)
                    else:
                        index_ltp = await asyncio.to_thread(self.dhan.get_index_ltp, index_name)
                        if index_ltp > 0:
                            bot_state['index_ltp'] = index_ltp
                
                # Paper testing: simulate index movement when bypass_market_hours is enabled.
                # This keeps candles moving even if the API returns stale/flat prices outside market hours.
                if bot_state.get('mode') == 'paper' and config.get('bypass_market_hours', False):
                    if bot_state.get('simulated_base_price') is None:
                        # Prefer any existing index_ltp as starting point; else fall back to a realistic base.
                        if bot_state.get('index_ltp', 0) > 0:
                            bot_state['simulated_base_price'] = float(bot_state['index_ltp'])
                        else:
                            if index_name == 'NIFTY':
                                bot_state['simulated_base_price'] = 23500.0
                            elif index_name == 'BANKNIFTY':
                                bot_state['simulated_base_price'] = 51500.0
                            elif index_name == 'FINNIFTY':
                                bot_state['simulated_base_price'] = 22000.0
                            elif index_name == 'MIDCPNIFTY':
                                bot_state['simulated_base_price'] = 12500.0
                            else:
                                bot_state['simulated_base_price'] = 70000.0  # SENSEX

                    tick_change = random.choice([-15, -10, -5, -2, 0, 2, 5, 10, 15])
                    bot_state['simulated_base_price'] += tick_change
                    bot_state['index_ltp'] = round(bot_state['simulated_base_price'], 2)
                    logger.debug(f"[TEST] Simulated {index_name} LTP: {bot_state['index_ltp']}")
                
                # Update candle data
                index_ltp = bot_state['index_ltp']
                if provider != 'mds' and index_ltp > 0:
                    if candle_open <= 0:
                        candle_open = float(index_ltp)
                    if index_ltp > high:
                        high = index_ltp
                    if index_ltp < low:
                        low = index_ltp
                    close = index_ltp

                    # Update HTF candle data (for sub-1m trading intervals)
                    if config.get('htf_filter_enabled', True) and config.get('candle_interval', 60) < 60:
                        if index_ltp > htf_high:
                            htf_high = index_ltp
                        if index_ltp < htf_low:
                            htf_low = index_ltp
                        htf_close = index_ltp

                # Check if HTF candle is complete
                if provider != 'mds' and config.get('htf_filter_enabled', True) and config.get('candle_interval', 60) < 60:
                    htf_seconds = int(config.get('htf_filter_timeframe', 60))
                    if htf_seconds != 60:
                        htf_seconds = 60

                    htf_elapsed = (datetime.now() - htf_candle_start_time).total_seconds()
                    if htf_elapsed >= htf_seconds:
                        htf_candle_number += 1
                        if htf_high > 0 and htf_low < float('inf'):
                            htf_value, htf_signal = self.htf_indicator.add_candle(htf_high, htf_low, htf_close)

                            if htf_value:
                                bot_state['htf_supertrend_value'] = htf_value if isinstance(htf_value, (int, float)) else str(htf_value)

                                if htf_signal == "GREEN":
                                    bot_state['htf_signal_status'] = "buy"
                                elif htf_signal == "RED":
                                    bot_state['htf_signal_status'] = "sell"
                                else:
                                    bot_state['htf_signal_status'] = "waiting"

                                if htf_signal:
                                    bot_state['htf_supertrend_signal'] = htf_signal

                                logger.info(
                                    f"[HTF CANDLE CLOSE #{htf_candle_number}] {index_name} | "
                                    f"H={htf_high:.2f} L={htf_low:.2f} C={htf_close:.2f} | "
                                    f"HTF_ST={htf_value:.2f} | "
                                    f"Signal={htf_signal or 'None'}"
                                )

                        htf_candle_start_time = datetime.now()
                        htf_high, htf_low, htf_close = 0, float('inf'), 0
                
                # Check SL/Target on EVERY TICK (responsive protection)
                if self.current_position and bot_state['current_option_ltp'] > 0:
                    option_ltp = bot_state['current_option_ltp']
                    tick_exit = await self.check_tick_sl(option_ltp)
                    if tick_exit:
                        # Position exited on tick, reset candle for next entry
                        candle_start_time = datetime.now()
                        candle_open = 0.0
                        high, low, close = 0, float('inf'), 0
                        candle_number = 0
                        await asyncio.sleep(1)
                        continue
                
                # Check if candle is complete
                elapsed = (datetime.now() - candle_start_time).total_seconds()
                if provider != 'mds' and elapsed >= candle_interval:
                    current_candle_time = datetime.now()
                    candle_number += 1

                    await self._handle_closed_candle(
                        index_name=index_name,
                        candle_number=candle_number,
                        candle_interval=int(config.get('candle_interval', candle_interval) or candle_interval),
                        candle_open=float(candle_open or close or bot_state.get('index_ltp') or 0.0),
                        high=high,
                        low=low,
                        close=close,
                        current_candle_time=current_candle_time,
                    )

                    # Reset candle for next period
                    candle_start_time = datetime.now()
                    candle_open = 0.0
                    high, low, close = 0, float('inf'), 0
                
                # Handle paper mode simulation
                if self.current_position:
                    security_id = self.current_position.get('security_id', '')
                    
                    if security_id.startswith('SIM_'):
                        strike = self.current_position.get('strike', 0)
                        option_type = self.current_position.get('option_type', '')
                        index_ltp = bot_state['index_ltp']
                        
                        if strike and index_ltp:
                            distance_from_atm = abs(index_ltp - strike)
                            
                            if option_type == 'CE':
                                intrinsic = max(0, index_ltp - strike)
                            else:
                                intrinsic = max(0, strike - index_ltp)
                            
                            atm_time_value = 150
                            time_decay_factor = max(0, 1 - (distance_from_atm / 500))
                            time_value = atm_time_value * time_decay_factor
                            
                            simulated_ltp = intrinsic + time_value
                            tick_movement = random.choice([-0.10, -0.05, 0, 0.05, 0.10])
                            simulated_ltp += tick_movement
                            
                            simulated_ltp = round(simulated_ltp / 0.05) * 0.05
                            simulated_ltp = max(0.05, round(simulated_ltp, 2))
                            
                            bot_state['current_option_ltp'] = simulated_ltp
                
                # Broadcast state update
                await self.broadcast_state()
                
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[ERROR] Trading loop exception: {e}")
                await asyncio.sleep(5)
    
    async def broadcast_state(self):
        """Broadcast current state to WebSocket clients"""
        from server import manager
        
        # GET cooldown state for broadcast
        cooldown_state = self.cooldown_mgr.get_state()
        
        await manager.broadcast({
            "type": "state_update",
            "data": {
                "index_ltp": bot_state['index_ltp'],
                "supertrend_signal": bot_state['last_supertrend_signal'],
                "supertrend_value": bot_state['supertrend_value'],
                "htf_supertrend_signal": bot_state.get('htf_supertrend_signal'),
                "htf_supertrend_value": bot_state.get('htf_supertrend_value', 0.0),
                "position": bot_state['current_position'],
                "entry_price": bot_state['entry_price'],
                "current_option_ltp": bot_state['current_option_ltp'],
                "trailing_sl": bot_state['trailing_sl'],
                "daily_pnl": bot_state['daily_pnl'],
                "daily_trades": bot_state['daily_trades'],
                "is_running": bot_state['is_running'],
                "mode": bot_state['mode'],
                "mds_score": bot_state.get('mds_score', 0.0),
                "mds_slope": bot_state.get('mds_slope', 0.0),
                "mds_acceleration": bot_state.get('mds_acceleration', 0.0),
                "mds_stability": bot_state.get('mds_stability', 0.0),
                "mds_confidence": bot_state.get('mds_confidence', 0.0),
                "mds_is_choppy": bot_state.get('mds_is_choppy', False),
                "mds_direction": bot_state.get('mds_direction', 'NONE'),
                "trading_enabled": bool(config.get('trading_enabled', True)),
                "selected_index": config['selected_index'],
                "candle_interval": config['candle_interval'],
                "htf_filter_enabled": bool(config.get('htf_filter_enabled', True)),
                "htf_filter_timeframe": int(config.get('htf_filter_timeframe', 60)),
                
                # NEW: Trading Logic v2 Metrics
                "market_regime": bot_state.get('market_regime', 'UNKNOWN'),
                "entry_confidence": self.entry_confidence_recorded,
                "runtime_confidence": self.runtime_confidence,
                "cooldown_active": cooldown_state.is_active,
                "cooldown_candles_remaining": cooldown_state.candles_remaining,
                "cooldown_reason": cooldown_state.reason.value if cooldown_state.reason else None,
                
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })

    async def process_mds_on_close(self, mds_snapshot, index_ltp: float) -> bool:
        """Process score-engine snapshot on candle close.

        Entry/exit are driven purely by MDS + safety gates.
        """
        exited = False
        index_name = config['selected_index']
        index_config = get_index_config(index_name)

        # Exit logic first
        if self.current_position:
            position_type = self.current_position.get('option_type', '')
            qty = int(self.current_position.get('qty') or 0)
            if qty <= 0:
                qty = int(config.get('order_qty', 1)) * index_config['lot_size']

            # Respect min-hold to avoid churn
            if self._min_hold_active():
                return False

            score = float(getattr(mds_snapshot, 'score', 0.0) or 0.0)
            slope = float(getattr(mds_snapshot, 'slope', 0.0) or 0.0)

            # Use slow timeframe MACD+Histogram as a confirmation signal for exits.
            # This reduces churn exits caused by base-TF noise.
            slow_mom = 0.0
            try:
                tf_scores = getattr(mds_snapshot, 'tf_scores', {}) or {}
                if isinstance(tf_scores, dict) and tf_scores:
                    slow_tf = max(int(k) for k in tf_scores.keys())
                    slow = tf_scores.get(slow_tf)
                    slow_mom = float(getattr(slow, 'macd_score', 0.0) or 0.0) + float(getattr(slow, 'hist_score', 0.0) or 0.0)
            except Exception:
                slow_mom = 0.0

            # Deterministic exits (score-only)
            neutral = abs(score) <= 6.0

            should_exit = False
            reason = ""
            if position_type == 'CE':
                if score <= -10.0:
                    if slow_mom <= -1.0:
                        should_exit = True
                        reason = "MDS Reversal (slow confirm)"
                elif neutral:
                    if abs(slow_mom) <= 1.0:
                        should_exit = True
                        reason = "MDS Neutral (slow confirm)"
                elif slope <= -2.0 and score < 12.0:
                    if slow_mom <= 0.0:
                        should_exit = True
                        reason = "MDS Momentum Loss (slow confirm)"
            elif position_type == 'PE':
                if score >= 10.0:
                    if slow_mom >= 1.0:
                        should_exit = True
                        reason = "MDS Reversal (slow confirm)"
                elif neutral:
                    if abs(slow_mom) <= 1.0:
                        should_exit = True
                        reason = "MDS Neutral (slow confirm)"
                elif slope >= 2.0 and score > -12.0:
                    if slow_mom >= 0.0:
                        should_exit = True
                        reason = "MDS Momentum Loss (slow confirm)"

            if should_exit:
                exit_price = bot_state['current_option_ltp']
                pnl = (exit_price - self.entry_price) * qty
                logger.warning(f"[MDS] ✗ EXIT | {position_type} | Score={score:.2f} Slope={slope:.2f} SlowMom={slow_mom:.1f} | Reason={reason} | P&L=₹{pnl:.2f}")
                closed = await self.close_position(exit_price, pnl, reason)
                exited = bool(closed)
                if closed:
                    self.last_signal = None

            return exited

        # No position: entry logic

        # Enforce order cooldown between any two orders
        if not self._can_place_new_entry_order():
            remaining = self._remaining_entry_cooldown()
            logger.info(f"[MDS] Entry blocked - Order cooldown active ({remaining:.1f}s remaining)")
            return False

        if not can_take_new_trade():
            logger.info("[ENTRY_DECISION] NO | Reason=after_cutoff (MDS)")
            return False

        if bot_state['daily_trades'] >= config['max_trades_per_day']:
            logger.info(f"[MDS] Max daily trades reached ({config['max_trades_per_day']})")
            return False

        # Check min_trade_gap protection (optional)
        min_gap = config.get('min_trade_gap', 0)
        if min_gap > 0 and self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last < min_gap:
                logger.info(f"[ENTRY_DECISION] NO | Reason=min_trade_gap (MDS) ({time_since_last:.1f}s < {min_gap}s)")
                return False

        # Score-engine gates
        if not getattr(mds_snapshot, 'ready', False):
            logger.info("[MDS] Skipping - Engine not ready yet (warming up)")
            return False

        if bool(getattr(mds_snapshot, 'is_choppy', False)):
            logger.info("[MDS] Skipping - Market is choppy")
            return False

        direction = str(getattr(mds_snapshot, 'direction', 'NONE') or 'NONE')
        score = float(getattr(mds_snapshot, 'score', 0.0) or 0.0)
        slope = float(getattr(mds_snapshot, 'slope', 0.0) or 0.0)
        confidence = float(getattr(mds_snapshot, 'confidence', 0.0) or 0.0)

        # Require meaningful score + slope
        if direction == 'NONE':
            logger.info(f"[ENTRY_DECISION] NO | Reason=neutral_band (MDS) | Score={score:.2f} Slope={slope:.2f}")
            self._mds_last_direction = direction
            self._mds_confirm_count = 0
            return False

        if abs(score) < 10.0:
            logger.info(f"[ENTRY_DECISION] NO | Reason=score_too_low (MDS) | Score={score:.2f} Slope={slope:.2f} Dir={direction}")
            self._mds_last_direction = direction
            self._mds_confirm_count = 0
            return False

        if abs(slope) < 1.0:
            logger.info(f"[ENTRY_DECISION] NO | Reason=slope_too_low (MDS) | Score={score:.2f} Slope={slope:.2f} Dir={direction}")
            self._mds_last_direction = direction
            self._mds_confirm_count = 0
            return False

        # Multi-candle confirmation
        # In paper mode, use a looser confirm requirement so simulations can actually place trades
        # under the synthetic price generator (which tends to be choppy).
        confirm_needed = 1 if bot_state.get('mode') == 'paper' else 2
        if self._mds_last_direction == direction:
            self._mds_confirm_count += 1
        else:
            self._mds_last_direction = direction
            self._mds_confirm_count = 1

        if self._mds_confirm_count < confirm_needed:
            logger.info(
                f"[MDS] Arming entry ({self._mds_confirm_count}/{confirm_needed}) | "
                f"Dir={direction} Score={score:.2f} Slope={slope:.2f} Conf={confidence:.2f}"
            )
            return False

        # Confidence sizing (capped by risk-per-trade)
        max_lots = int(config.get('order_qty', 1) or 1)
        sizer = PositionSizingAgent(max_lots=max_lots)
        sizing = sizer.size(
            confidence=confidence,
            risk_per_trade_rupees=float(config.get('risk_per_trade', 0) or 0),
            sl_points=float(config.get('initial_stoploss', 0) or 0),
            lot_size=int(index_config['lot_size']),
        )

        if sizing.final_lots < 1:
            logger.info(f"[MDS] Skipping - Confidence sizing resulted in 0 lots | Conf={confidence:.2f}")
            return False

        option_type = 'CE' if direction == 'CE' else 'PE'
        atm_strike = round_to_strike(index_ltp, index_name)

        logger.info(
            f"[MDS] ENTRY | {option_type} | Score={score:.2f} Slope={slope:.2f} Conf={confidence:.2f} | "
            f"Lots={sizing.final_lots} (Desired={sizing.desired_lots}, RiskCap={sizing.risk_cap_lots}) | "
            f"Index={index_name} LTP={index_ltp:.2f} ATM={atm_strike}"
        )

        before = bot_state.get('current_position')
        await self.enter_position(option_type, atm_strike, index_ltp, override_lots=int(sizing.final_lots))
        after = bot_state.get('current_position')
        if before is None and after is not None:
            logger.info(f"[ENTRY_DECISION] YES | Confirmed (MDS) | {option_type} {atm_strike}")
        else:
            logger.info(f"[ENTRY_DECISION] NO | Entry blocked downstream (MDS) | {option_type} {atm_strike}")

        self.last_trade_time = datetime.now()
        self._mds_confirm_count = 0
        return False
    
    async def check_trailing_sl(self, current_ltp: float):
        """Update SL values - initial fixed SL then trails profit using step-based method"""
        if not self.current_position:
            return
        
        # Check if trailing is completely disabled
        trail_start = config.get('trail_start_profit', 0)
        trail_step = config.get('trail_step', 0)
        
        if trail_start == 0 or trail_step == 0:
            # Trailing disabled - don't set any SL
            return
        
        profit_points = current_ltp - self.entry_price
        
        # Track highest profit reached
        if profit_points > self.highest_profit:
            self.highest_profit = profit_points
        
        # Step 1: Set initial fixed stoploss (if enabled)
        initial_sl = config.get('initial_stoploss', 0)
        if initial_sl > 0 and self.trailing_sl is None:
            self.trailing_sl = self.entry_price - initial_sl
            bot_state['trailing_sl'] = self.trailing_sl
            logger.info(f"[SL] Initial SL set: {self.trailing_sl:.2f} ({initial_sl} pts below entry)")
            return
        
        # Step 2: Start trailing SL after reaching trail_start_profit
        
        # Only start trailing after profit reaches trail_start_profit
        if profit_points < trail_start:
            return
        
        # Trailing behavior:
        # - Start trailing only after profit reaches trail_start
        # - Once active, SL trails behind highest profit by trail_step in step increments
        # Example: entry=100, start=10, step=5
        #   at profit=10 (LTP=110) -> SL locks +5 (105)
        #   at profit=15 (LTP=115) -> SL locks +10 (110)
        trail_levels = int((self.highest_profit - trail_start) / trail_step)
        locked_profit = (trail_start - trail_step) + (trail_levels * trail_step)
        locked_profit = max(0.0, locked_profit)
        new_sl = self.entry_price + locked_profit
        
        # Always move SL up, never down (protect profit)
        if self.trailing_sl is None or new_sl > self.trailing_sl:
            old_sl = self.trailing_sl
            self.trailing_sl = new_sl
            bot_state['trailing_sl'] = self.trailing_sl
            
            if old_sl and old_sl > (self.entry_price - initial_sl):
                # This is a trailing update (not initial trigger)
                logger.info(f"[SL] Trailing SL updated: {old_sl:.2f} → {new_sl:.2f} (Profit: {profit_points:.2f} pts)")
            else:
                # This is the first trailing activation
                logger.info(f"[SL] Trailing started: {new_sl:.2f} (Profit: {profit_points:.2f} pts)")

    
    async def check_trailing_sl_on_close(self, current_ltp: float) -> bool:
        """Check if trailing SL or target is hit on candle close"""
        if not self.current_position:
            return False
        
        index_config = get_index_config(config['selected_index'])
        qty = int(self.current_position.get('qty') or 0)
        if qty <= 0:
            qty = config['order_qty'] * index_config['lot_size']
        profit_points = current_ltp - self.entry_price
        
        # Check target first (if enabled)
        target_points = config.get('target_points', 0)
        if target_points > 0 and profit_points >= target_points:
            pnl = profit_points * qty
            logger.info(
                f"[EXIT] Target hit | LTP={current_ltp:.2f} | Entry={self.entry_price:.2f} | Profit={profit_points:.2f} pts | Target={target_points:.2f} pts"
            )
            closed = await self.close_position(current_ltp, pnl, "Target Hit")
            return bool(closed)
        
        # Update trailing SL
        await self.check_trailing_sl(current_ltp)
        
        # Check trailing SL
        if self.trailing_sl and current_ltp <= self.trailing_sl:
            pnl = (current_ltp - self.entry_price) * qty
            logger.info(f"[EXIT] Trailing SL hit | LTP={current_ltp:.2f} | SL={self.trailing_sl:.2f}")
            closed = await self.close_position(current_ltp, pnl, "Trailing SL Hit")
            return bool(closed)
        
        return False
    
    async def check_tick_sl(self, current_ltp: float) -> bool:
        """Check SL/Target on every tick (more responsive than candle close)"""
        if not self.current_position:
            return False
        
        index_config = get_index_config(config['selected_index'])
        qty = int(self.current_position.get('qty') or 0)
        if qty <= 0:
            qty = config['order_qty'] * index_config['lot_size']
        profit_points = current_ltp - self.entry_price
        pnl = profit_points * qty
        
        # Check DAILY max loss FIRST (highest priority)
        daily_max_loss = config.get('daily_max_loss', 0)
        if daily_max_loss > 0 and bot_state['daily_pnl'] + pnl < -daily_max_loss:
            logger.warning(
                f"[EXIT] ✗ Daily max loss BREACHED! | Current Daily P&L=₹{bot_state['daily_pnl']:.2f} | This trade P&L=₹{pnl:.2f} | Limit=₹{-daily_max_loss:.2f} | FORCE SQUAREOFF"
            )
            closed = await self.close_position(current_ltp, pnl, "Daily Max Loss")
            if closed:
                bot_state['daily_max_loss_triggered'] = True
            return bool(closed)
        
        # Check max loss per trade (if enabled)
        max_loss_per_trade = config.get('max_loss_per_trade', 0)
        if max_loss_per_trade > 0 and pnl < -max_loss_per_trade:
            logger.info(
                f"[EXIT] Max loss per trade hit | LTP={current_ltp:.2f} | Entry={self.entry_price:.2f} | Loss=₹{abs(pnl):.2f} | Limit=₹{max_loss_per_trade:.2f}"
            )
            closed = await self.close_position(current_ltp, pnl, "Max Loss Per Trade")
            return bool(closed)

        # Check target (if enabled)
        target_points = config.get('target_points', 0)
        if target_points > 0 and profit_points >= target_points:
            logger.info(
                f"[EXIT] Target hit (tick) | LTP={current_ltp:.2f} | Entry={self.entry_price:.2f} | Profit={profit_points:.2f} pts | Target={target_points:.2f} pts"
            )
            closed = await self.close_position(current_ltp, pnl, "Target Hit")
            return bool(closed)
        
        # Update trailing SL values
        await self.check_trailing_sl(current_ltp)
        
        # Check if trailing SL is breached
        if self.trailing_sl and current_ltp <= self.trailing_sl:
            pnl = (current_ltp - self.entry_price) * qty
            logger.info(f"[EXIT] Trailing SL hit (tick) | LTP={current_ltp:.2f} | SL={self.trailing_sl:.2f}")
            closed = await self.close_position(current_ltp, pnl, "Trailing SL Hit")
            return bool(closed)
        
        return False
    
    async def process_signal_on_close(self, signal: str, index_ltp: float, flipped: bool = False) -> bool:
        """Process SuperTrend signal on candle close"""
        exited = False
        index_name = config['selected_index']
        index_config = get_index_config(index_name)
        qty = 0
        if self.current_position:
            qty = int(self.current_position.get('qty') or 0)
        if qty <= 0:
            qty = config['order_qty'] * index_config['lot_size']
        
        # Check for exit on SuperTrend direction reversal (PRIMARY exit trigger)
        # Exit based on SuperTrend direction change - this is the critical signal
        if self.current_position:
            position_type = self.current_position.get('option_type', '')
            # Get current SuperTrend direction from indicator
            st_direction = getattr(self.indicator, 'direction', 0)
            
            # CRITICAL: EXIT ON SIGNAL REVERSAL
            if position_type == 'CE' and st_direction == -1:  # Holding CE but ST flipped RED
                if self._min_hold_active():
                    logger.info("[SIGNAL] Reversal detected but min-hold active - exit blocked")
                else:
                    exit_price = bot_state['current_option_ltp']
                    pnl = (exit_price - self.entry_price) * qty
                    logger.warning(f"[SIGNAL] ✗ REVERSAL: SuperTrend flipped RED - Exiting CE position IMMEDIATELY | P&L=₹{pnl:.2f}")
                    closed = await self.close_position(exit_price, pnl, "SuperTrend Reversal")
                    exited = bool(closed)
                    if closed:
                        # Clear last_signal to allow re-entry on opposite signal (RED)
                        self.last_signal = None
                    # Continue to enter opposite position (PE)
            
            elif position_type == 'PE' and st_direction == 1:  # Holding PE but ST flipped GREEN
                if self._min_hold_active():
                    logger.info("[SIGNAL] Reversal detected but min-hold active - exit blocked")
                else:
                    exit_price = bot_state['current_option_ltp']
                    pnl = (exit_price - self.entry_price) * qty
                    logger.warning(f"[SIGNAL] ✗ REVERSAL: SuperTrend flipped GREEN - Exiting PE position IMMEDIATELY | P&L=₹{pnl:.2f}")
                    closed = await self.close_position(exit_price, pnl, "SuperTrend Reversal")
                    exited = bool(closed)
                    if closed:
                        # Clear last_signal to allow re-entry on opposite signal (GREEN)
                        self.last_signal = None
                    # Continue to enter opposite position (CE)
        
        # Check if new trade allowed
        if self.current_position:
            logger.info("[ENTRY_DECISION] NO | Reason=position_open")
            return exited

        # Enforce order cooldown between any two orders (prevents exit->entry flip too fast)
        if not self._can_place_new_entry_order():
            remaining = self._remaining_entry_cooldown()
            logger.info(f"[ENTRY] ✗ Skipping - Order cooldown active ({remaining:.1f}s remaining)")
            logger.info("[ENTRY_DECISION] NO | Reason=order_cooldown")
            return exited
        
        if not can_take_new_trade():
            logger.info("[ENTRY_DECISION] NO | Reason=after_cutoff")
            return exited
        
        if bot_state['daily_trades'] >= config['max_trades_per_day']:
            logger.info(f"[SIGNAL] Max daily trades reached ({config['max_trades_per_day']})")
            logger.info("[ENTRY_DECISION] NO | Reason=max_daily_trades")
            return exited
        
        # Check min_trade_gap protection (optional)
        min_gap = config.get('min_trade_gap', 0)
        if min_gap > 0 and self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last < min_gap:
                logger.info(f"[ENTRY_DECISION] NO | Reason=min_trade_gap ({time_since_last:.1f}s < {min_gap}s)")
                return exited
        
        # Enter new position
        if not signal:
            logger.info("[ENTRY_DECISION] NO | Reason=no_signal")
            return exited

        # Trade only on SuperTrend FLIP (candle-to-candle), not based on previous trade state
        if config.get('trade_only_on_flip', False) and not flipped:
            logger.info(f"[ENTRY] ✗ Skipping - No SuperTrend flip this candle | Signal={signal}")
            logger.info("[ENTRY_DECISION] NO | Reason=no_flip")
            return exited

        # MACD CONFIRMATION: require MACD line to confirm direction at candle close
        if config.get('macd_confirmation_enabled', True):
            if not self.macd or self.macd.last_macd is None or self.macd.last_signal_line is None:
                logger.info("[ENTRY] ✗ Skipping - MACD not ready yet")
                logger.info("[ENTRY_DECISION] NO | Reason=macd_not_ready")
                return exited

            eps = 1e-9
            diff = float(self.macd.last_macd) - float(self.macd.last_signal_line)
            bullish = diff >= -eps
            if signal == 'GREEN' and not bullish:
                logger.info(
                    f"[ENTRY] ✗ Skipping - MACD not confirming BUY | "
                    f"MACD={self.macd.last_macd:.4f} SIG={self.macd.last_signal_line:.4f}"
                )
                logger.info("[ENTRY_DECISION] NO | Reason=macd_not_confirming_buy")
                return exited
            if signal == 'RED' and bullish:
                logger.info(
                    f"[ENTRY] ✗ Skipping - MACD not confirming SELL | "
                    f"MACD={self.macd.last_macd:.4f} SIG={self.macd.last_signal_line:.4f}"
                )
                logger.info("[ENTRY_DECISION] NO | Reason=macd_not_confirming_sell")
                return exited

        # MTF FILTER: If trading below 1m, only take entries aligned with 1m SuperTrend direction
        if config.get('htf_filter_enabled', True) and config.get('candle_interval', 60) < 60:
            htf_direction = getattr(self.htf_indicator, 'direction', 0) if self.htf_indicator else 0

            required = 1 if signal == 'GREEN' else -1
            if htf_direction == 0:
                logger.info("[ENTRY] ✗ Skipping - HTF SuperTrend not ready yet (need 1m candles)")
                logger.info("[ENTRY_DECISION] NO | Reason=htf_not_ready")
                return exited

            if htf_direction != required:
                htf_side = 'GREEN' if htf_direction == 1 else 'RED'
                logger.info(f"[ENTRY] ✗ Skipping - HTF filter mismatch | LTF={signal}, HTF={htf_side}")
                logger.info("[ENTRY_DECISION] NO | Reason=htf_mismatch")
                return exited
        
        # NOTE: Previously we compared against last trade signal (self.last_signal).
        # That behavior is replaced by candle-level flip detection via the `flipped` flag.
        
        option_type = 'PE' if signal == 'RED' else 'CE'
        atm_strike = round_to_strike(index_ltp, index_name)
        
        # Log signal details
        logger.info(
            f"[ENTRY] Taking {option_type} | {signal} Signal | "
            f"Index: {index_name} | "
            f"LTP: {index_ltp:.2f} | "
            f"ATM Strike: {atm_strike} | "
            f"SuperTrend: {bot_state['supertrend_value']:.2f}"
        )

        before = bot_state.get('current_position')
        await self.enter_position(option_type, atm_strike, index_ltp)
        after = bot_state.get('current_position')
        if before is None and after is not None:
            logger.info(f"[ENTRY_DECISION] YES | Confirmed | {option_type} {atm_strike}")
        else:
            logger.info(f"[ENTRY_DECISION] NO | Entry blocked downstream | {option_type} {atm_strike}")
        self.last_trade_time = datetime.now()
        
        return exited
    
    async def enter_position(self, option_type: str, strike: int, index_ltp: float, override_lots: int | None = None):
        """Enter a new position with market validation"""
        # Soft pause: keep bot running (prices/indicators/exits), but block new entries
        if not config.get('trading_enabled', True):
            now = datetime.now(timezone.utc)
            if (
                self._last_entries_paused_log_time is None
                or (now - self._last_entries_paused_log_time).total_seconds() >= 10
            ):
                logger.info(
                    f"[ENTRY] Skipped - Trading paused (trading_enabled=false) | Would take {option_type} {strike}"
                )
                self._last_entries_paused_log_time = now
            return

        # CRITICAL: Double-check market is open before entering
        if not is_market_open():
            logger.warning(f"[ENTRY] ✗ BLOCKED - Market is CLOSED | Cannot enter {option_type} position")
            return
        
        # CHECK: Trading hours protection
        if not self.is_within_trading_hours():
            logger.warning(f"[ENTRY] ✗ BLOCKED - Outside trading hours | Cannot enter {option_type} position")
            return

        # Enforce minimum cooldown between orders
        if not self._can_place_new_entry_order():
            remaining = self._remaining_entry_cooldown()
            logger.info(f"[ENTRY] ✗ BLOCKED - Order cooldown active ({remaining:.1f}s remaining)")
            return
        
        index_name = config['selected_index']
        index_config = get_index_config(index_name)

        lots = int(config.get('order_qty', 1) or 1)
        if override_lots is not None:
            lots = max(1, int(override_lots))

        # Calculate position size based on risk (if enabled)
        risk_per_trade = float(config.get('risk_per_trade', 0) or 0)
        sl_points = float(config.get('initial_stoploss', 0) or 0)
        if risk_per_trade > 0 and sl_points > 0:
            max_lots = int(risk_per_trade / (sl_points * index_config['lot_size']))
            if max_lots < 1:
                logger.warning(
                    f"[POSITION] ✗ BLOCKED - risk_per_trade too low for 1 lot | Risk=₹{risk_per_trade} SL={sl_points} LotSize={index_config['lot_size']}"
                )
                return
            lots = max(1, min(int(max_lots), int(lots)))
            qty = lots * index_config['lot_size']
            logger.info(
                f"[POSITION] Size adjusted for risk: {lots} lots ({qty} qty) (Risk: ₹{risk_per_trade}, SL: {sl_points}pts)"
            )
        else:
            qty = lots * index_config['lot_size']
        
        trade_id = f"T{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Get expiry
        expiry = await self.dhan.get_nearest_expiry(index_name) if self.dhan else None
        if not expiry:
            ist = get_ist_time()
            expiry_day = index_config['expiry_day']
            days_until_expiry = (expiry_day - ist.weekday()) % 7
            if days_until_expiry == 0 and ist.hour >= 15:
                days_until_expiry = 7
            expiry_date = ist + timedelta(days=days_until_expiry)
            expiry = expiry_date.strftime("%Y-%m-%d")
        
        entry_price = 0
        security_id = ""
        
        # Paper mode (fully simulated pricing; avoids mixing simulated index candles with live option quotes)
        if bot_state['mode'] == 'paper':
            security_id = f"SIM_{index_name}_{strike}_{option_type}"
            
            if entry_price <= 0:
                distance = abs(index_ltp - strike)
                intrinsic = max(0, index_ltp - strike) if option_type == 'CE' else max(0, strike - index_ltp)
                time_value = 150 * max(0, 1 - (distance / 500))
                entry_price = round((intrinsic + time_value) / 0.05) * 0.05
                entry_price = round(entry_price, 2)
            
            logger.info(
                f"[ENTRY] PAPER | {index_name} {option_type} {strike} | Expiry: {expiry} | Price: {entry_price} | Qty: {qty}"
            )
        
        # Live mode
        else:
            if not self.dhan:
                logger.error("[ERROR] Dhan API not initialized")
                return

            # Resolve instrument + a more accurate entry price from market data
            try:
                security_id = await self.dhan.get_atm_option_security_id(index_name, strike, option_type, expiry)
                if not security_id:
                    logger.error(f"[ERROR] Could not find security ID for {index_name} {strike} {option_type}")
                    return

                option_ltp = await self.dhan.get_option_ltp(
                    security_id=security_id,
                    strike=strike,
                    option_type=option_type,
                    expiry=expiry,
                    index_name=index_name
                )
                if option_ltp > 0:
                    entry_price = round(option_ltp / 0.05) * 0.05
                    entry_price = round(entry_price, 2)
            except Exception as e:
                logger.error("[ERROR] Failed to get entry price: %s", e)
                return
            
            result = await self.dhan.place_order(security_id, "BUY", qty, index_name=index_name)
            logger.info(f"[ORDER] Entry order result: {result}")
            
            # Check if order was successfully placed
            if result.get('status') != 'success' or not result.get('orderId'):
                logger.error(f"[ERROR] Failed to place entry order: {result}")
                return
            
            # Order placed successfully - save to DB immediately
            order_id = result.get('orderId')

            # Track last order timestamp (entry order) right after placing the order
            self.last_order_time_utc = datetime.now(timezone.utc)

            verify = await self.dhan.verify_order_filled(
                order_id=str(order_id),
                security_id=str(security_id),
                expected_qty=int(qty),
                timeout_seconds=30
            )
            if not verify.get('filled'):
                logger.error(
                    f"[ORDER] ✗ Entry order NOT filled | OrderID: {order_id} | Status: {verify.get('status')} | {verify.get('message')}"
                )
                return

            avg_price = float(verify.get('average_price') or 0)
            if avg_price > 0:
                entry_price = round(avg_price / 0.05) * 0.05
                entry_price = round(entry_price, 2)
            
            logger.info(
                f"[ENTRY] LIVE | {index_name} {option_type} {strike} | Expiry: {expiry} | OrderID: {order_id} | Fill Price: {entry_price} | Qty: {qty}"
            )

        # Track last order timestamp (paper mode entry)
        if bot_state['mode'] == 'paper':
            self.last_order_time_utc = datetime.now(timezone.utc)
        
        # Save position
        self.current_position = {
            'trade_id': trade_id,
            'option_type': option_type,
            'strike': strike,
            'expiry': expiry,
            'security_id': security_id,
            'index_name': index_name,
            'qty': qty,
            'entry_time': datetime.now(timezone.utc).isoformat()
        }
        self.entry_price = entry_price
        self.trailing_sl = None
        self.highest_profit = 0
        self.entry_time_utc = datetime.now(timezone.utc)
        
        bot_state['current_position'] = self.current_position
        bot_state['entry_price'] = self.entry_price
        bot_state['daily_trades'] += 1
        bot_state['current_option_ltp'] = entry_price
        
        # ONLY set last_signal AFTER position is successfully confirmed open
        self.last_signal = option_type[0].upper() + 'E'  # 'CE' -> 'C', 'PE' -> 'P'
        self.last_signal = 'GREEN' if option_type == 'CE' else 'RED'
        
        # Save to database in background - don't wait for DB commit
        asyncio.create_task(save_trade({
            'trade_id': trade_id,
            'entry_time': datetime.now(timezone.utc).isoformat(),
            'option_type': option_type,
            'strike': strike,
            'expiry': expiry,
            'entry_price': self.entry_price,
            'qty': qty,
            'mode': bot_state['mode'],
            'index_name': index_name,
            'created_at': datetime.now(timezone.utc).isoformat()
        }))

