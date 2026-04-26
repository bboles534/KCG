# Neon FinBrain

`Neon FinBrain` is a Python-first financial intelligence platform built to feel closer to an AI market copilot than a basic stock dashboard.

It combines:

- real-time market monitoring on free, no-key data feeds
- technical indicators and local machine learning predictions
- news sentiment and credibility scoring
- portfolio, budget, and expense tracking
- paper trading and what-if scenario analysis
- a context-aware AI chat assistant with memory

## Default stack

The app runs out of the box with free sources and no API keys:

- market data: `yfinance`
- ticker news: Yahoo Finance + Google News RSS
- analytics: local Python signal engine
- ML: local `scikit-learn` random forest model
- storage: local SQLite
- UI + API: `NiceGUI` + `FastAPI`

Optional connectors are supported if you later want to wire them in:

- Robinhood login sync
- Plaid transaction sync
- Polygon market data
- Alpaca market data

## Features

- Live market engine refreshing every 3 seconds by default
- Intraday candles, spread tracking, volume spike detection, and volatility scoring
- RSI, MACD, moving averages, ATR, pattern detection, and trend classification
- AI decision engine with confidence, risk labels, and reasoning
- News sentiment plus basic fake-news / rumor heuristics using source credibility
- Portfolio tracker with budget advice and risk heatmap
- Paper trading simulator
- What-if scenario engine
- Floating AI chat assistant with memory stored in SQLite
- JSON API endpoints for market, brain, portfolio, and chat

## Project layout

```text
app/
  main.py                  NiceGUI app + FastAPI endpoints
  config.py                environment-driven settings
  schemas.py               shared pydantic models
  store.py                 SQLite persistence
  services/
    market_data.py         free/default + optional paid market providers
    analytics.py           indicators and feature engineering
    ml_engine.py           local ML predictor
    news.py                Yahoo + RSS news aggregation
    sentiment.py           sentiment and credibility scoring
    portfolio.py           manual + optional connector portfolio tracking
    brain.py               decision support and strategy synthesis
    chat.py                memory-aware AI chat logic
    simulator.py           paper trading
    scenarios.py           stress testing
    orchestrator.py        app state orchestration
data/
  financial_brain.db       created automatically
```

## Run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Optionally copy `.env.example` to `.env` and adjust the watchlist or refresh rate.
4. Start the platform:

```bash
python -m app.main
```

5. Open [http://localhost:8080](http://localhost:8080)

## API routes

- `GET /api/state`
- `GET /api/market`
- `GET /api/brain`
- `POST /api/chat`
- `POST /api/portfolio/positions`
- `POST /api/portfolio/expenses`

Example:

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What are the best day-trade setups right now?"}'
```

## Configuration

Important environment variables:

- `REFRESH_SECONDS`
- `WATCHLIST`
- `MARKET_INTERVAL`
- `MARKET_PERIOD`
- `MAX_CHART_POINTS`
- `MAX_NEWS_ITEMS`

Optional connectors:

- `ROBINHOOD_USERNAME`
- `ROBINHOOD_PASSWORD`
- `PLAID_CLIENT_ID`
- `PLAID_SECRET`
- `PLAID_ACCESS_TOKEN`
- `PLAID_BASE_URL`
- `POLYGON_API_KEY`
- `ALPACA_API_KEY`
- `ALPACA_API_SECRET`

## Notes

- The default no-key mode uses live public market feeds, but quote freshness depends on the upstream source and may be delayed.
- Plaid and premium market APIs are optional; the platform is designed to stay fully usable without them.
- The existing legacy static files in the workspace are not used by this application entrypoint.
