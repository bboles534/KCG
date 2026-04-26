# Neon FinBrain — Full Financial Dashboard

Single-file financial intelligence platform with RL trading terminal.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in your keys
cp .env.example .env
# Edit .env with your Gemini API key (free at aistudio.google.com)

# 3. Run
python3 neon_finbrain.py

# 4. Open browser
# http://localhost:8080
```

## What's Included

| Feature | Status | Notes |
|---|---|---|
| Live market data (yfinance) | ✅ Free, no key needed |
| Gemini AI chatbot | ✅ Needs GEMINI_API_KEY |
| Portfolio tracker | ✅ Manual + Robinhood sync |
| Plaid bank transactions | ✅ Mock mode enabled by default |
| News + sentiment | ✅ Free RSS feeds |
| ML price predictions | ✅ Local scikit-learn |
| Paper trading simulator | ✅ Built-in |
| RL Trading Terminal | ✅ PPO agent, needs stable-baselines3 |
| Scenario stress-testing | ✅ Built-in |

## What You Need From Your Side

1. **Gemini API Key** (required for AI chat)
   - Go to https://aistudio.google.com
   - Click "Get API Key" → Create key
   - Paste into `.env` as `GEMINI_API_KEY=...`
   - Free tier is sufficient

2. **Nothing else** — everything else runs out of the box

## Optional Extras

- **Plaid real data**: Run the two curl commands in the docs to get an access_token, set `PLAID_MOCK=false`
- **Robinhood sync**: Install `pip install robin_stocks`, add credentials to `.env`
- **Alpaca data**: Already in `.env` from your keys above

## RL Trader Usage

1. Scroll to the **RL Trading Terminal** panel
2. Pick a ticker, date range, and risk profile
3. Click **▶ Train & Backtest**
4. Results appear inline — agent return, alpha vs buy & hold, trade log

First run downloads model weights (~few seconds). Training takes 5–30 seconds depending on timesteps.
