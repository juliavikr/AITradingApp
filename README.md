# AI Trading Bot

A desktop GUI trading bot built with Python, Tkinter, and the Alpaca Trade API.

This application manages a simple automated trading system that:
- Enters a position using a market order
- Averages down using laddered limit buy orders
- Tracks and cancels **only** bot-owned orders safely
- Persists configuration and state locally
- Optionally uses an AI assistant to analyze portfolio risk and open orders

⚠️ **WARNING**  
This bot implements an averaging-down strategy. There is **no exit logic, no stop loss, no position sizing, and no portfolio-level risk management**. Running this live without extending it is dangerous. Paper trade only unless you fully understand and modify the system.

---

## Screenshot

![AI Trading Bot UI](./docs/screenshot.png)

---

## Features

- **GUI**: Add symbols, set `Levels` + `Drawdown %`, view positions and bot status.
- **Three-state control per symbol**:
  - `Inactive` → no trading activity
  - `Armed` → ready, but will not place orders until you explicitly place
  - `Active` → bot manages entry + ladder orders on each tick
- **Bot-safe order tagging** using `client_order_id` prefix: `BOT:<SYMBOL>:...`
- **Persistence**: saves configured symbols/levels/status to `equities.json`
- **Optional AI Assistant** panel (OpenAI) to comment on risk, holdings, and open orders

---

## Tech Stack

- Python 3
- Tkinter
- Alpaca Trade API
- python-dotenv
- OpenAI SDK

---

## Requirements

- Python 3.9+
- Tkinter (bundled with Python on most platforms)
- Alpaca account (paper trading recommended)

Python dependencies:
- `alpaca-trade-api`
- `python-dotenv`
- `openai`

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install alpaca-trade-api python-dotenv openai
```

### 4. Environment configuration

Create a `.env` file in the project root:

```env
ALPACA_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

OPENAI_SECRET_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
```

The application will not start if Alpaca credentials are missing.

---

## Running the App

```bash
python app.py
```

---

## GUI Usage Guide

### Adding an Equity

**Fields:**
- **Symbol** – Stock ticker (e.g. AAPL)
- **Levels** – Number of averaging-down levels (max 20)
- **Drawdown %** – Percentage drop per level (min 0.5%)
- **Tick (s)** – Bot loop interval in seconds

**When added:**
- Entry price is estimated from Alpaca market data
- Ladder prices are precomputed
- Symbol is stored with status `Inactive`

---

## Symbol States

### Inactive
- Stored only
- No trading logic applied

### Armed
- Approved for trading
- No orders placed yet

### Active
- Entry order ensured
- Ladder orders actively managed
- Evaluated on every tick

---

## Placing Orders

1. Select one or more **Armed** symbols
2. Click **Place Orders**
3. Confirm the dialog

**Result:**
- Symbol transitions to `Active`
- Trading logic executes immediately
- Continues running on every tick

---

## Deactivating Symbols

- Cancels only bot-owned orders
- Uses `client_order_id` prefix matching
- Positions remain open
- Status returns to `Armed`

---

## Removing Symbols

**Options:**
- Remove only (leave broker orders untouched)
- Remove and cancel bot orders
- Cancel removal

---

## Trading Algorithm

### Background Loop

Runs every `tick_seconds`:

1. Collect all symbols with status `Active`
2. Check market open via Alpaca clock
3. Fetch open broker orders
4. For each active symbol:
   - Ensure entry position exists
   - Recompute ladder prices from actual filled entry
   - Submit missing limit orders
   - Prevent duplicates using client IDs

---

### Entry Logic

If no position exists:
- Submit market buy (qty = 1)
- `client_order_id = BOT:<SYMBOL>:ENTRY`
- Poll until filled or terminal
- Use filled average price as entry reference

---

### Ladder Generation

For each level `i`:

```
price = round(entry_price * (1 - drawdown * i), 2)
```

**Constraints enforced:**
- Levels ≤ 20
- Drawdown ≥ 0.5%
- Minimum price spacing: $0.05
- No negative or zero prices

---

### Level State Tracking

- Positive level key → pending
- Negative level key → order submitted

**Example:**
```json
{
  "1": 23.20,
  "2": 23.09,
  "-3": 22.97
}
```

---

## Order Safety

All bot orders use deterministic `client_order_id`s:

- **Entry:** `BOT:<SYMBOL>:ENTRY`
- **Level:** `BOT:<SYMBOL>:L<LEVEL>`

This guarantees:
- No duplicate submissions
- Safe cancellation
- No interference with manual or external orders

---

## Persistence

**File:** `equities.json`

Stores per symbol:
- Entry price
- Levels count
- Drawdown
- Level state
- Status (`Inactive`, `Armed`, `Active`)

---

## Known Limitations

- No sell or exit logic
- No stop losses
- Fixed order size (1 share)
- No volatility-based spacing
- No max risk or exposure limits

---

## Disclaimer

This software is provided for educational purposes only. You are solely responsible for any orders submitted to your broker. Use at your own risk.