import os
import tkinter as tk
from tkinter import ttk, messagebox
import json
import threading
import time
from datetime import datetime

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError
from dotenv import load_dotenv

load_dotenv()

DATA_FILE = "equities.json"

ALPACA_KEY = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

OPENAI_KEY = os.getenv("OPENAI_SECRET_KEY")

if not ALPACA_KEY or not ALPACA_SECRET:
    raise RuntimeError("Missing ALPACA_KEY or ALPACA_SECRET_KEY in environment.")

api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, BASE_URL, api_version="v2")

MAX_LEVELS = 20
MIN_DRAWDOWN_PCT = 0.5         # minimum allowed drawdown input in percent
MIN_PRICE_STEP = 0.05          # minimum spacing between levels in dollars (after rounding)

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def now_ts():
    return datetime.now().strftime("%H:%M:%S")


def market_is_open():
    try:
        clock = api.get_clock()
        return bool(getattr(clock, "is_open", False))
    except Exception:
        return False


def get_position_or_none(symbol):
    try:
        return api.get_position(symbol)
    except APIError as e:
        msg = str(e).lower()
        if "position does not exist" in msg or "404" in msg:
            return None
        raise


def wait_for_order_fill(order_id, timeout=180, poll=1.0):
    end = time.time() + timeout
    while time.time() < end:
        o = api.get_order(order_id)
        status = getattr(o, "status", None)
        if status == "filled":
            return o
        if status in ("canceled", "rejected", "expired"):
            return o
        time.sleep(poll)
    return api.get_order(order_id)


def get_latest_price(symbol):
    try:
        t = api.get_latest_trade(symbol)
        p = safe_float(getattr(t, "price", None))
        if p and p > 0:
            return p
    except Exception:
        pass

    try:
        q = api.get_latest_quote(symbol)
        bid = safe_float(getattr(q, "bid_price", None))
        ask = safe_float(getattr(q, "ask_price", None))
        if bid and ask and bid > 0 and ask > 0:
            return (bid + ask) / 2.0
    except Exception:
        pass

    return None


def fetch_positions_map():
    out = {}
    try:
        positions = api.list_positions()
    except Exception:
        return out

    for pos in positions:
        out[pos.symbol] = {
            "qty": safe_float(pos.qty, 0) or 0,
            "avg_entry_price": safe_float(getattr(pos, "avg_entry_price", None), 0) or 0,
            "current_price": safe_float(getattr(pos, "current_price", None), 0) or 0,
            "unrealized_pl": safe_float(getattr(pos, "unrealized_pl", None), 0) or 0,
        }
    return out


def fetch_open_orders():
    try:
        orders = api.list_orders(status="open", limit=500)
    except Exception:
        return []
    out = []
    for o in orders:
        out.append({
            "id": getattr(o, "id", None),
            "symbol": getattr(o, "symbol", ""),
            "side": getattr(o, "side", ""),
            "type": getattr(o, "type", ""),
            "qty": safe_float(getattr(o, "qty", None), 0) or 0,
            "limit_price": safe_float(getattr(o, "limit_price", None), 0) or 0,
            "client_order_id": getattr(o, "client_order_id", None),
        })
    return out

def bot_prefix(symbol: str) -> str:
    return f"BOT:{symbol}:"


def client_id_entry(symbol: str) -> str:
    # entry market buy
    return f"{bot_prefix(symbol)}ENTRY"


def client_id_level(symbol: str, level: int) -> str:
    return f"{bot_prefix(symbol)}L{level}"


def is_bot_order_for_symbol(order, symbol: str) -> bool:
    cid = order.get("client_order_id")
    return isinstance(cid, str) and cid.startswith(bot_prefix(symbol))


def open_orders_by_client_id(open_orders):
    s = set()
    for o in open_orders:
        cid = o.get("client_order_id")
        if cid:
            s.add(cid)
    return s


def chatgpt_response(message, portfolio_data, open_orders):
    pre_prompt = f"""
You are an AI portfolio analyst.

Tasks:
1) Assess risk exposure of current holdings.
2) Analyze open orders and their potential impact.
3) Evaluate portfolio health and diversification.
4) Highlight key market risks relevant to this portfolio.
5) Suggest practical risk management adjustments.

Portfolio data:
{portfolio_data}

Open orders:
{open_orders}

User question:
{message}
""".strip()

    if not OPENAI_KEY:
        return "(OpenAI error) Missing OPENAI_SECRET_KEY in environment."

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": pre_prompt},
                {"role": "user", "content": message},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception as e_new:
        try:
            import openai
            openai.api_key = OPENAI_KEY
            resp = openai.ChatCompletion.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": pre_prompt},
                    {"role": "user", "content": message},
                ],
                temperature=0.2,
            )
            return resp["choices"][0]["message"]["content"]
        except Exception as e_old:
            return f"(OpenAI error) new_sdk={e_new} | legacy={e_old}"


STATUS_INACTIVE = "Inactive"
STATUS_ARMED = "Armed"
STATUS_ACTIVE = "Active"


class TradingBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Trading Bot")
        self.root.minsize(980, 620)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        self.root.rowconfigure(5, weight=1)

        self.equities = self.load_equities()
        self.running = True
        self.lock = threading.Lock()

        self._tick_seconds = 5.0
        self._last_market_closed_log = 0.0

        # Header
        header = tk.Frame(root)
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        header.columnconfigure(0, weight=1)

        tk.Label(header, text="AI Trading Bot", font=("TkDefaultFont", 14, "bold")).grid(row=0, column=0, sticky="w")
        self.account_var = tk.StringVar(value="Account: —")
        tk.Label(header, textvariable=self.account_var, fg="#555").grid(row=0, column=1, sticky="e")

        # Add Equity Form
        self.form_frame = tk.LabelFrame(root, text="Add Equity", padx=10, pady=10)
        self.form_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=6)
        for c in range(9):
            self.form_frame.columnconfigure(c, weight=0)
        self.form_frame.columnconfigure(8, weight=1)

        tk.Label(self.form_frame, text="Symbol").grid(row=0, column=0, sticky="w")
        self.symbol_entry = tk.Entry(self.form_frame, width=10)
        self.symbol_entry.grid(row=0, column=1, sticky="w", padx=(6, 14))

        tk.Label(self.form_frame, text="Levels").grid(row=0, column=2, sticky="w")
        self.levels_entry = tk.Entry(self.form_frame, width=6)
        self.levels_entry.grid(row=0, column=3, sticky="w", padx=(6, 14))

        tk.Label(self.form_frame, text="Drawdown %").grid(row=0, column=4, sticky="w")
        self.drawdown_entry = tk.Entry(self.form_frame, width=8)
        self.drawdown_entry.grid(row=0, column=5, sticky="w", padx=(6, 14))

        self.add_button = tk.Button(self.form_frame, text="Add", command=self.add_equity, width=10)
        self.add_button.grid(row=0, column=6, sticky="w")

        tk.Label(self.form_frame, text="Tick (s)").grid(row=0, column=7, sticky="e")
        self.tick_entry = tk.Entry(self.form_frame, width=6)
        self.tick_entry.insert(0, "5")
        self.tick_entry.grid(row=0, column=8, sticky="w", padx=(6, 0))

        self.symbol_entry.bind("<Return>", lambda e: self.add_equity())
        self.drawdown_entry.bind("<Return>", lambda e: self.add_equity())

        # Table
        table_frame = tk.Frame(root)
        table_frame.grid(row=2, column=0, sticky="nsew", padx=12, pady=6)
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        cols = ("Symbol", "Qty", "Avg Entry", "Last", "Unreal. P/L", "Levels", "Status")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="extended")
        for col in cols:
            self.tree.heading(col, text=col)

        self.tree.column("Symbol", width=90, anchor="w")
        self.tree.column("Qty", width=70, anchor="e")
        self.tree.column("Avg Entry", width=95, anchor="e")
        self.tree.column("Last", width=85, anchor="e")
        self.tree.column("Unreal. P/L", width=95, anchor="e")
        self.tree.column("Levels", width=380, anchor="w")
        self.tree.column("Status", width=95, anchor="center")

        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns", padx=(6, 0))

        # Double click = Arm
        self.tree.bind("<Double-1>", lambda e: self.arm_selected())

        # Controls (no more confusing extra cancel button)
        controls = tk.Frame(root)
        controls.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 6))
        controls.columnconfigure(6, weight=1)

        tk.Button(controls, text="Arm", command=self.arm_selected, width=10).grid(row=0, column=0, padx=(0, 8))
        tk.Button(controls, text="Place Orders", command=self.place_orders_selected, width=14).grid(row=0, column=1, padx=(0, 8))
        tk.Button(controls, text="Deactivate (Cancel Bot Orders)", command=self.deactivate_selected, width=26).grid(row=0, column=2, padx=(0, 8))
        tk.Button(controls, text="Remove", command=self.remove_selected_equity, width=10).grid(row=0, column=3, padx=(0, 8))

        self.status_var = tk.StringVar(value="Idle")
        tk.Label(controls, textvariable=self.status_var, fg="#555").grid(row=0, column=6, sticky="e")

        # Log panel
        log_box = tk.LabelFrame(root, text="Bot Log", padx=10, pady=10)
        log_box.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 6))
        log_box.columnconfigure(0, weight=1)

        self.log_text = tk.Text(log_box, height=6, state=tk.DISABLED, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="ew")

        # Chat
        chat_box = tk.LabelFrame(root, text="AI Assistant", padx=10, pady=10)
        chat_box.grid(row=5, column=0, sticky="nsew", padx=12, pady=(6, 12))
        chat_box.rowconfigure(1, weight=1)
        chat_box.columnconfigure(0, weight=1)

        input_row = tk.Frame(chat_box)
        input_row.grid(row=0, column=0, sticky="ew")
        input_row.columnconfigure(0, weight=1)

        self.chat_input = tk.Entry(input_row)
        self.chat_input.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.chat_input.bind("<Return>", lambda e: self.send_message())

        self.send_button = tk.Button(input_row, text="Ask", command=self.send_message, width=10)
        self.send_button.grid(row=0, column=1)

        output_frame = tk.Frame(chat_box)
        output_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        self.chat_output = tk.Text(output_frame, height=8, state=tk.DISABLED, wrap="word")
        chat_scroll = ttk.Scrollbar(output_frame, orient="vertical", command=self.chat_output.yview)
        self.chat_output.configure(yscrollcommand=chat_scroll.set)
        self.chat_output.grid(row=0, column=0, sticky="nsew")
        chat_scroll.grid(row=0, column=1, sticky="ns", padx=(6, 0))

        # Keep tick cached updated
        self.sync_controls()

        # Periodic UI refresh (replaces refresh button)
        self.periodic_ui_refresh()

        # Start trading thread
        self.auto_update_thread = threading.Thread(target=self.auto_update, daemon=True)
        self.auto_update_thread.start()

        self.symbol_entry.focus_set()

    def sync_controls(self):
        try:
            tick = float(self.tick_entry.get().strip())
            self._tick_seconds = max(1.0, tick)
        except Exception:
            self._tick_seconds = 5.0
        self.root.after(500, self.sync_controls)

    def periodic_ui_refresh(self):
        self.refresh_all()
        self.root.after(2000, self.periodic_ui_refresh)

    def ui_set_status(self, text):
        self.root.after(0, lambda: self.status_var.set(text))

    def log(self, msg):
        def _append():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"[{now_ts()}] {msg}\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        self.root.after(0, _append)

    def _levels_summary(self, levels_dict):
        if not isinstance(levels_dict, dict) or not levels_dict:
            return "—"
        pending = sorted([k for k in levels_dict.keys() if isinstance(k, int) and k > 0])
        placed = sorted([k for k in levels_dict.keys() if isinstance(k, int) and k < 0])

        def fmt_keys(keys):
            if not keys:
                return "0"
            return f"{len(keys)} ({', '.join(map(str, keys[:4]))}{'…' if len(keys) > 4 else ''})"

        return f"Pending: {fmt_keys(pending)} | Placed: {fmt_keys(placed)}"

    def save_equities(self):
        with self.lock:
            to_save = {}
            for sym, data in self.equities.items():
                d = dict(data)
                levels = d.get("levels", {})
                if isinstance(levels, dict):
                    d["levels"] = {str(k): v for k, v in levels.items()}
                to_save[sym] = d

        with open(DATA_FILE, "w") as f:
            json.dump(to_save, f, indent=2)

    def load_equities(self):
        try:
            with open(DATA_FILE, "r") as f:
                raw = json.load(f)

            fixed = {}
            for sym, data in raw.items():
                d = dict(data)

                levels = d.get("levels", {})
                if isinstance(levels, dict):
                    new_levels = {}
                    for k, v in levels.items():
                        try:
                            ik = int(k)
                        except Exception:
                            continue
                        new_levels[ik] = v
                    d["levels"] = new_levels

                d.setdefault("levels_count", 5)
                d.setdefault("status", STATUS_INACTIVE)
                d.setdefault("drawdown", 0.02)
                fixed[sym] = d

            return fixed
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def build_levels(self, entry_price: float, levels_count: int, drawdown_pct: float):
        """
        Returns {level:int -> price:float} with:
        - levels_count capped
        - drawdown >= MIN_DRAWDOWN_PCT
        - monotonic decreasing
        - min spacing enforced after rounding
        """
        levels_count = int(levels_count)
        if levels_count <= 0:
            raise ValueError("Levels must be positive.")
        if levels_count > MAX_LEVELS:
            raise ValueError(f"Levels too high. Max is {MAX_LEVELS}.")

        if drawdown_pct < MIN_DRAWDOWN_PCT:
            raise ValueError(f"Drawdown too small. Min is {MIN_DRAWDOWN_PCT}%.")

        dd = drawdown_pct / 100.0

        levels = {}
        last_price = None
        for i in range(1, levels_count + 1):
            price = round(entry_price * (1 - dd * i), 2)
            if price <= 0:
                raise ValueError("Drawdown too large: level price went <= 0.")

            if last_price is not None:
                if (last_price - price) < MIN_PRICE_STEP:
                    raise ValueError(
                        f"Levels too tight. Increase drawdown or reduce levels. "
                        f"(Min spacing {MIN_PRICE_STEP:.2f} violated around level {i}.)"
                    )
                if price >= last_price:
                    raise ValueError("Internal error: level prices not decreasing.")

            levels[i] = price
            last_price = price

        return levels

    # ----------------------------
    # Selection helpers
    # ----------------------------

    def _selected_symbols(self):
        items = self.tree.selection()
        if not items:
            return []
        syms = []
        for item in items:
            vals = self.tree.item(item)["values"]
            if vals:
                syms.append(vals[0])
        return syms

    # ----------------------------
    # UI actions
    # ----------------------------

    def add_equity(self):
        symbol = self.symbol_entry.get().strip().upper()
        levels_raw = self.levels_entry.get().strip()
        drawdown_raw = self.drawdown_entry.get().strip()

        if not symbol:
            messagebox.showerror("Invalid Input", "Symbol is required (e.g., AAPL).")
            return
        if not levels_raw.isdigit() or int(levels_raw) <= 0:
            messagebox.showerror("Invalid Input", f"Levels must be a positive integer (max {MAX_LEVELS}).")
            return

        try:
            drawdown_pct = float(drawdown_raw)
        except Exception:
            messagebox.showerror("Invalid Input", "Drawdown must be a number (e.g., 2.5).")
            return

        entry_price = get_latest_price(symbol)
        if not entry_price:
            entry_price = 100.0

        try:
            levels_count = int(levels_raw)
            level_prices = self.build_levels(entry_price, levels_count, drawdown_pct)
        except Exception as e:
            messagebox.showerror("Invalid Strategy", str(e))
            return

        with self.lock:
            self.equities[symbol] = {
                "entry_price": float(entry_price),
                "levels_count": int(levels_raw),
                "levels": level_prices,      # pending levels are positive keys
                "drawdown": float(drawdown_pct) / 100.0,
                "status": STATUS_INACTIVE,
            }

        self.save_equities()
        self.log(f"Added {symbol} ({STATUS_INACTIVE}), entry est {entry_price:.2f}")
        self.ui_set_status(f"Added {symbol}")

        self.symbol_entry.focus_set()
        self.symbol_entry.select_range(0, tk.END)

    def arm_selected(self):
        symbols = self._selected_symbols()
        if not symbols:
            messagebox.showwarning("No Selection", "Select at least one equity.")
            return

        changed = 0
        blocked = []
        with self.lock:
            for sym in symbols:
                cur = self.equities.get(sym, {}).get("status", STATUS_INACTIVE)
                if cur == STATUS_ACTIVE:
                    blocked.append(sym)
                    continue
                if cur != STATUS_ARMED:
                    self.equities[sym]["status"] = STATUS_ARMED
                    changed += 1

        if blocked:
            self.log(f"Arm blocked (Active): {', '.join(blocked)} — Deactivate first.")

        if changed:
            self.save_equities()
            self.log(f"Armed: {', '.join(symbols)}")
            self.refresh_all()
        elif not blocked:
            messagebox.showinfo("No Change", "Nothing changed.")

    def place_orders_selected(self):
        symbols = self._selected_symbols()
        if not symbols:
            messagebox.showwarning("No Selection", "Select at least one equity.")
            return

        with self.lock:
            eligible = [s for s in symbols if self.equities.get(s, {}).get("status") == STATUS_ARMED]

        if not eligible:
            messagebox.showinfo("Not Armed", "Select equities with status 'Armed' to place orders.")
            return

        if not messagebox.askyesno(
            "Confirm Place Orders",
            "This will submit broker orders for the selected equities.\nProceed?"
        ):
            return

        with self.lock:
            for sym in eligible:
                self.equities[sym]["status"] = STATUS_ACTIVE

        self.save_equities()
        self.refresh_all()

        def _run():
            self.ui_set_status("Placing orders...")
            try:
                self.trade_systems(symbol_filter=set(eligible))
            finally:
                self.ui_set_status("Idle")
                self.root.after(0, self.refresh_all)

        threading.Thread(target=_run, daemon=True).start()

    def deactivate_selected(self):
        symbols = self._selected_symbols()
        if not symbols:
            messagebox.showwarning("No Selection", "Select at least one equity.")
            return

        if not messagebox.askyesno(
            "Confirm Deactivate",
            "This will cancel ONLY the bot's open orders for selected symbols\n"
            "and set their status to Armed.\n\n"
            "Positions will NOT be closed.\nProceed?"
        ):
            return

        def _run():
            self.ui_set_status("Deactivating...")
            total_cancelled = 0
            for sym in symbols:
                total_cancelled += self.cancel_bot_orders_for_symbol(sym)

            with self.lock:
                for sym in symbols:
                    if sym in self.equities:
                        self.equities[sym]["status"] = STATUS_ARMED

            self.save_equities()
            self.log(f"Deactivated: {', '.join(symbols)} (cancelled {total_cancelled} bot orders)")
            self.ui_set_status("Idle")
            self.root.after(0, self.refresh_all)

        threading.Thread(target=_run, daemon=True).start()

    def remove_selected_equity(self):
        symbols = self._selected_symbols()
        if not symbols:
            messagebox.showwarning("No Selection", "Select at least one equity.")
            return

        # Two explicit choices — no surprises
        choice = messagebox.askyesnocancel(
            "Remove Options",
            "Yes = Remove + cancel bot orders at broker\n"
            "No  = Remove only (leave broker orders untouched)\n"
            "Cancel = abort"
        )
        if choice is None:
            return

        def _run():
            cancelled = 0
            if choice is True:
                self.ui_set_status("Removing + cancelling...")
                for sym in symbols:
                    cancelled += self.cancel_bot_orders_for_symbol(sym)
            else:
                self.ui_set_status("Removing...")

            with self.lock:
                for sym in symbols:
                    self.equities.pop(sym, None)

            self.save_equities()
            if choice is True:
                self.log(f"Removed: {', '.join(symbols)} (cancelled {cancelled} bot orders)")
            else:
                self.log(f"Removed: {', '.join(symbols)} (broker orders untouched)")
            self.ui_set_status("Idle")
            self.root.after(0, self.refresh_all)

        threading.Thread(target=_run, daemon=True).start()

    def cancel_bot_orders_for_symbol(self, symbol: str) -> int:
        """
        Cancels ONLY orders whose client_order_id starts with BOT:<symbol>:
        This prevents nuking unrelated orders and explains cancellations precisely.
        """
        open_orders = fetch_open_orders()
        cancelled = 0
        for o in open_orders:
            if o.get("id") and is_bot_order_for_symbol(o, symbol):
                try:
                    api.cancel_order(o["id"])
                    cancelled += 1
                except Exception as e:
                    self.log(f"Cancel failed {symbol} {o['id']}: {e}")
        return cancelled

    def send_message(self):
        message = self.chat_input.get().strip()
        if not message:
            return

        self.send_button.config(state=tk.DISABLED)
        self.ui_set_status("Assistant: thinking...")

        def _run():
            positions = fetch_positions_map()
            open_orders = fetch_open_orders()

            portfolio_data = []
            for sym, p in positions.items():
                portfolio_data.append({
                    "symbol": sym,
                    "qty": p["qty"],
                    "entry_price": p["avg_entry_price"],
                    "current_price": p["current_price"],
                    "unrealized_pl": p["unrealized_pl"],
                })

            response = chatgpt_response(message, portfolio_data, open_orders)

            def _render():
                self.chat_output.config(state=tk.NORMAL)
                self.chat_output.insert(tk.END, f"You: {message}\n{response}\n\n")
                self.chat_output.see(tk.END)
                self.chat_output.config(state=tk.DISABLED)
                self.chat_input.delete(0, tk.END)
                self.send_button.config(state=tk.NORMAL)
                self.ui_set_status("Idle")

            self.root.after(0, _render)

        threading.Thread(target=_run, daemon=True).start()

    # ----------------------------
    # Trading logic (only Active symbols)
    # ----------------------------

    def place_limit_level(self, symbol, price, level, open_client_ids, submitted_client_ids):
        cid = client_id_level(symbol, level)

        # Prevent duplicates in this tick AND across broker state
        if cid in submitted_client_ids or cid in open_client_ids:
            return

        try:
            api.submit_order(
                symbol=symbol,
                qty=1,
                side="buy",
                type="limit",
                time_in_force="gtc",
                limit_price=round(price, 2),
                client_order_id=cid,
            )
            submitted_client_ids.add(cid)

            # mark placed locally
            with self.lock:
                levels_map = self.equities.get(symbol, {}).get("levels", {})
                levels_map[-level] = round(price, 2)
                if level in levels_map:
                    del levels_map[level]

            self.log(f"Placed LIMIT BUY {symbol} @ {price:.2f} (level {level})")
        except Exception as e:
            self.log(f"Order error {symbol} level {level}: {e}")

    def trade_systems(self, symbol_filter=None):
        with self.lock:
            items = list(self.equities.items())

        active = []
        for sym, data in items:
            if data.get("status") != STATUS_ACTIVE:
                continue
            if symbol_filter is not None and sym not in symbol_filter:
                continue
            active.append((sym, data))

        if not active:
            return

        if not market_is_open():
            # log at most once per 60 seconds
            now = time.time()
            if now - self._last_market_closed_log > 60:
                self._last_market_closed_log = now
                self.log("Market closed; skipping trading this tick.")
            return

        open_orders = fetch_open_orders()
        open_client_ids = open_orders_by_client_id(open_orders)
        submitted_client_ids = set()  # prevent dupes within one loop

        for symbol, data in active:
            # Ensure position exists (entry)
            try:
                pos = get_position_or_none(symbol)
            except Exception as e:
                self.log(f"get_position error {symbol}: {e}")
                continue

            if pos is None:
                entry_cid = client_id_entry(symbol)
                if entry_cid in open_client_ids:
                    # entry order already open
                    continue

                try:
                    order = api.submit_order(
                        symbol=symbol,
                        qty=1,
                        side="buy",
                        type="market",
                        time_in_force="day",
                        client_order_id=entry_cid,
                    )
                    self.log(f"Submitted MARKET BUY {symbol} (order {order.id})")
                except Exception as e:
                    self.log(f"Initial order failed {symbol}: {e}")
                    continue

                try:
                    filled = wait_for_order_fill(order.id, timeout=180, poll=1.0)
                except Exception as e:
                    self.log(f"Order poll failed {symbol} {order.id}: {e}")
                    continue

                if getattr(filled, "status", None) != "filled":
                    self.log(f"Entry not filled {symbol}. Status={getattr(filled, 'status', None)}")
                    continue

                try:
                    pos = api.get_position(symbol)
                except Exception as e:
                    self.log(f"Filled but position fetch failed {symbol}: {e}")
                    continue

            entry_price = safe_float(getattr(pos, "avg_entry_price", None), None)
            if not entry_price or entry_price <= 0:
                self.log(f"Invalid entry price {symbol}: {entry_price}")
                continue

            drawdown = safe_float(data.get("drawdown", 0), 0) or 0
            levels_count = int(data.get("levels_count", 1))

            # Recompute targets based on real entry price
            targets = {i: round(entry_price * (1 - drawdown * i), 2) for i in range(1, levels_count + 1)}

            with self.lock:
                levels_dict = self.equities[symbol].get("levels", {})
                if not isinstance(levels_dict, dict):
                    levels_dict = {}

                # merge missing pending levels
                for level, price in targets.items():
                    if level not in levels_dict and -level not in levels_dict:
                        levels_dict[level] = price

                self.equities[symbol]["entry_price"] = float(entry_price)
                self.equities[symbol]["levels"] = levels_dict

            # Place pending levels
            for level, price in targets.items():
                with self.lock:
                    cur_levels = self.equities[symbol]["levels"]
                    if level not in cur_levels:
                        continue  # already placed/removed

                self.place_limit_level(symbol, price, level, open_client_ids, submitted_client_ids)

        self.save_equities()

    # ----------------------------
    # UI refresh
    # ----------------------------

    def refresh_account(self):
        try:
            acct = api.get_account()
            cash = safe_float(getattr(acct, "cash", None), 0) or 0
            equity = safe_float(getattr(acct, "equity", None), 0) or 0
            last_eq = safe_float(getattr(acct, "last_equity", None), None)
            day_pl = (equity - last_eq) if (last_eq is not None) else None
            if day_pl is None:
                self.account_var.set(f"Account: cash {cash:.2f} | equity {equity:.2f}")
            else:
                self.account_var.set(f"Account: cash {cash:.2f} | equity {equity:.2f} | day P/L {day_pl:.2f}")
        except Exception:
            self.account_var.set("Account: —")

    def refresh_table(self):
        positions = fetch_positions_map()

        for row in self.tree.get_children():
            self.tree.delete(row)

        with self.lock:
            items = list(self.equities.items())

        for symbol, data in items:
            p = positions.get(symbol, {})
            qty = p.get("qty", 0) or 0
            avg_entry = p.get("avg_entry_price", data.get("entry_price", 0) or 0) or 0
            last = p.get("current_price", get_latest_price(symbol) or 0) or 0
            upl = p.get("unrealized_pl", 0) or 0

            levels_text = self._levels_summary(data.get("levels"))
            self.tree.insert(
                "",
                "end",
                values=(
                    symbol,
                    f"{qty:.4g}" if qty else "0",
                    f"{avg_entry:.2f}" if avg_entry else "—",
                    f"{last:.2f}" if last else "—",
                    f"{upl:.2f}" if upl else "0.00",
                    levels_text,
                    data.get("status", STATUS_INACTIVE),
                ),
            )

    def refresh_all(self):
        self.refresh_account()
        self.refresh_table()

    # ----------------------------
    # Background loop
    # ----------------------------

    def auto_update(self):
        while self.running:
            time.sleep(self._tick_seconds)
            self.ui_set_status("Running...")

            try:
                self.trade_systems()
            except Exception as e:
                self.log(f"trade_systems error: {e}")

            self.ui_set_status("Idle")

    def on_close(self):
        self.running = False
        self.save_equities()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = TradingBotGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
