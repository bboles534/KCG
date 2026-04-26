from __future__ import annotations

import asyncio

from fastapi.encoders import jsonable_encoder
from nicegui import app, ui
import plotly.graph_objects as go

from app.config import settings
from app.schemas import ExpenseRequest, PositionRequest
from app.services.orchestrator import PlatformOrchestrator

platform = PlatformOrchestrator(settings)


def _inject_theme() -> None:
    ui.add_head_html(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Unbounded:wght@500;700&display=swap" rel="stylesheet">
        """
    )
    ui.add_css(
        """
        :root {
          --bg: #060816;
          --panel: rgba(13, 17, 35, 0.72);
          --panel-strong: rgba(16, 22, 44, 0.88);
          --line: rgba(147, 220, 255, 0.18);
          --text: #edf5ff;
          --muted: #9cb2d6;
          --cyan: #4cf3ff;
          --green: #6dffac;
          --gold: #ffc857;
          --pink: #ff5dd3;
          --danger: #ff6b8a;
          --shadow: 0 25px 60px rgba(0, 0, 0, 0.36);
        }

        body {
          margin: 0;
          color: var(--text);
          font-family: 'Space Grotesk', sans-serif;
          background:
            radial-gradient(circle at top left, rgba(76, 243, 255, 0.14), transparent 24rem),
            radial-gradient(circle at top right, rgba(255, 93, 211, 0.12), transparent 22rem),
            linear-gradient(180deg, #040612 0%, #070b18 100%);
        }

        .page-shell {
          width: min(1440px, calc(100% - 2rem));
          margin: 0 auto;
          padding: 1.25rem 0 7rem;
          gap: 1rem;
        }

        .hero-card,
        .glass-card,
        .metric-card,
        .chat-card {
          background: var(--panel);
          border: 1px solid var(--line);
          backdrop-filter: blur(18px);
          box-shadow: var(--shadow);
          border-radius: 24px;
        }

        .hero-card {
          padding: 1.4rem;
          background:
            linear-gradient(135deg, rgba(76, 243, 255, 0.08), rgba(255, 93, 211, 0.08)),
            var(--panel-strong);
        }

        .glass-card {
          padding: 1rem;
        }

        .metric-card {
          padding: 1rem;
          min-height: 112px;
        }

        .metric-title,
        .subtle {
          color: var(--muted);
        }

        .title-font {
          font-family: 'Unbounded', sans-serif;
        }

        .neon-pill {
          border: 1px solid var(--line);
          border-radius: 999px;
          padding: 0.35rem 0.7rem;
          background: rgba(255, 255, 255, 0.04);
          color: var(--cyan);
          font-size: 0.78rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }

        .positive {
          color: var(--green);
        }

        .negative {
          color: var(--danger);
        }

        .neutral {
          color: var(--gold);
        }

        .ai-badge {
          background: linear-gradient(135deg, rgba(76, 243, 255, 0.16), rgba(255, 93, 211, 0.18));
          border-radius: 16px;
          padding: 0.65rem 0.8rem;
          border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .news-item,
        .insight-item,
        .chat-bubble {
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 18px;
          padding: 0.8rem;
          background: rgba(255, 255, 255, 0.03);
        }

        .chat-card {
          width: 380px;
          max-width: calc(100vw - 1rem);
          padding: 1rem;
        }

        .typing {
          animation: pulse 1.1s ease-in-out infinite;
        }

        @keyframes pulse {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 1; }
        }

        @media (max-width: 1100px) {
          .chat-card {
            position: static !important;
            width: 100%;
          }
        }
        """
    )


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _percent(value: float) -> str:
    return f"{value:.2f}%"


def _signed_percent(value: float) -> str:
    return f"{value:+.2f}%"


def _signed_money(value: float) -> str:
    return f"{value:+,.2f}"


def _color_class(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"


def _serialize_state():
    state = platform.get_state()
    return {
        "profile": state.profile.model_dump(),
        "market": {ticker: snapshot.model_dump() for ticker, snapshot in state.market.items()},
        "news": {
            ticker: [item.model_dump() for item in items]
            for ticker, items in state.news.items()
        },
        "portfolio": state.portfolio.model_dump(),
        "brain": state.brain.model_dump(),
        "simulation": state.simulation.model_dump(),
        "scenario": state.scenario.model_dump() if state.scenario else None,
        "refreshed_at": state.refreshed_at.isoformat(),
    }


def _price_figure(snapshot) -> go.Figure:
    figure = go.Figure()
    candles = snapshot.candles[-90:]
    if candles:
        x = [candle.timestamp for candle in candles]
        figure.add_trace(
            go.Candlestick(
                x=x,
                open=[candle.open for candle in candles],
                high=[candle.high for candle in candles],
                low=[candle.low for candle in candles],
                close=[candle.close for candle in candles],
                name=snapshot.ticker,
                increasing_line_color="#6dffac",
                decreasing_line_color="#ff6b8a",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=[snapshot.technicals.sma_20] * len(x),
                mode="lines",
                line={"color": "#4cf3ff", "width": 1.2, "dash": "dot"},
                name="SMA20 proxy",
            )
        )
    figure.update_layout(
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#edf5ff", "family": "Space Grotesk"},
        xaxis={"showgrid": False},
        yaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.08)"},
        showlegend=False,
        height=360,
    )
    return figure


def _risk_heatmap_figure(state) -> go.Figure:
    positions = state.portfolio.positions
    figure = go.Figure()
    if positions:
        figure.add_trace(
            go.Bar(
                x=[position.ticker for position in positions],
                y=[position.risk_score * 100 for position in positions],
                marker={"color": [position.allocation for position in positions], "colorscale": "Turbo"},
                text=[f"{position.allocation:.0%}" for position in positions],
                textposition="outside",
            )
        )
    figure.update_layout(
        title="Portfolio Risk Heatmap",
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#edf5ff", "family": "Space Grotesk"},
        xaxis={"showgrid": False},
        yaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.08)", "title": "Risk score"},
        height=310,
    )
    return figure


def _simulation_figure(state) -> go.Figure:
    figure = go.Figure()
    positions = state.simulation.positions
    if positions:
        figure.add_trace(
            go.Bar(
                x=[position.ticker for position in positions],
                y=[position.pnl_unrealized for position in positions],
                marker_color=[
                    "#6dffac" if position.pnl_unrealized >= 0 else "#ff6b8a"
                    for position in positions
                ],
            )
        )
    figure.update_layout(
        title="Paper Trading PnL",
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#edf5ff", "family": "Space Grotesk"},
        xaxis={"showgrid": False},
        yaxis={"showgrid": True, "gridcolor": "rgba(255,255,255,0.08)"},
        height=280,
    )
    return figure


@app.get("/api/state")
async def api_state():
    await platform.ensure_fresh()
    return jsonable_encoder(_serialize_state())


@app.get("/api/market")
async def api_market():
    await platform.ensure_fresh()
    return jsonable_encoder({ticker: snapshot.model_dump() for ticker, snapshot in platform.get_state().market.items()})


@app.get("/api/brain")
async def api_brain():
    await platform.ensure_fresh()
    return jsonable_encoder(platform.get_state().brain.model_dump())


@app.post("/api/chat")
async def api_chat(payload: dict):
    message = payload.get("message", "")
    reply, _ = await platform.chat(message)
    return jsonable_encoder(reply.model_dump())


@app.post("/api/portfolio/positions")
async def api_add_position(request: PositionRequest):
    state = await platform.add_position(request.ticker, request.shares, request.avg_cost, request.thesis)
    return jsonable_encoder({"portfolio": state.portfolio.model_dump()})


@app.post("/api/portfolio/expenses")
async def api_add_expense(request: ExpenseRequest):
    state = await platform.add_expense(request.category, request.amount, request.merchant)
    return jsonable_encoder({"portfolio": state.portfolio.model_dump()})


@ui.page("/")
async def home() -> None:
    _inject_theme()
    ui.dark_mode().enable()
    await platform.ensure_bootstrapped()
    state = platform.get_state()

    selected_ticker = {"value": state.profile.watchlist[0] if state.profile.watchlist else "AAPL"}

    def refresh_all_sections() -> None:
        hero_metrics.refresh()
        market_strip.refresh()
        chart_panel.refresh()
        ai_panel.refresh()
        portfolio_panel.refresh()
        news_panel.refresh()
        simulator_panel.refresh()
        chat_history.refresh()

    async def timed_refresh() -> None:
        await platform.refresh_all()
        refresh_all_sections()

    async def save_profile() -> None:
        watchlist = [ticker.strip().upper() for ticker in watchlist_input.value.split(",") if ticker.strip()]
        await platform.update_profile(
            horizon_select.value,
            risk_select.value,
            float(budget_input.value or 0),
            float(cash_input.value or 0),
            watchlist or settings.watchlist,
        )
        refresh_all_sections()
        ui.notify("Profile updated")

    async def save_position() -> None:
        await platform.add_position(
            position_ticker.value,
            float(position_shares.value or 0),
            float(position_cost.value or 0),
            position_thesis.value or "",
        )
        refresh_all_sections()
        ui.notify("Position saved")

    async def save_expense() -> None:
        await platform.add_expense(
            expense_category.value,
            float(expense_amount.value or 0),
            expense_merchant.value or "",
        )
        refresh_all_sections()
        ui.notify("Expense logged")

    async def execute_trade(side: str) -> None:
        await platform.execute_sim_trade(sim_ticker.value, side, float(sim_shares.value or 0))
        refresh_all_sections()
        ui.notify(f"Paper trade submitted: {side}")

    async def update_scenario() -> None:
        await platform.run_scenario(float(scenario_slider.value))
        simulator_panel.refresh()

    async def send_chat(prompt: str | None = None) -> None:
        message = prompt or chat_input.value or ""
        message = message.strip()
        if not message:
            return
        typing_note.set_visibility(True)
        reply, _ = await platform.chat(message)
        chat_input.set_value("")
        typing_note.set_visibility(False)
        refresh_all_sections()
        ui.notify("AI insight updated")
        if reply.updated_profile is not None:
            current_state = platform.get_state()
            horizon_select.set_value(current_state.profile.investment_horizon)
            risk_select.set_value(current_state.profile.risk_tolerance)
            budget_input.set_value(current_state.profile.monthly_budget)
            cash_input.set_value(current_state.profile.cash_balance)
            watchlist_input.set_value(", ".join(current_state.profile.watchlist))

    with ui.column().classes("page-shell"):
        with ui.row().classes("w-full items-center justify-between gap-4 hero-card"):
            with ui.column().classes("gap-2"):
                ui.label("Financial Intelligence Platform").classes("neon-pill")
                ui.label("Neon FinBrain").classes("title-font text-4xl")
                ui.label(
                    "A real-time AI market brain with no-key default data feeds, local ML, news credibility scoring, paper trading, and portfolio reasoning."
                ).classes("subtle max-w-3xl")
            with ui.column().classes("items-end gap-2"):
                ui.label("No-key mode: Yahoo Finance + free RSS/news").classes("neon-pill")
                ui.label("Optional connectors: Robinhood, Plaid, Polygon, Alpaca").classes("subtle")

        @ui.refreshable
        def hero_metrics() -> None:
            current = platform.get_state()
            with ui.grid(columns=5).classes("w-full gap-4"):
                metrics = [
                    ("Total Value", _money(current.portfolio.total_value), "metric-card"),
                    ("Day PnL", _money(current.portfolio.day_pnl), f"metric-card {_color_class(current.portfolio.day_pnl)}"),
                    ("Market Regime", current.brain.market_regime, "metric-card"),
                    ("Brain Confidence", f"{current.brain.confidence:.0%}", "metric-card"),
                    ("Last Refresh", current.refreshed_at.strftime("%H:%M:%S"), "metric-card"),
                ]
                for title, value, classes in metrics:
                    with ui.card().classes(classes):
                        ui.label(title).classes("metric-title")
                        ui.label(str(value)).classes("text-2xl font-bold")

        @ui.refreshable
        def market_strip() -> None:
            current = platform.get_state()
            with ui.grid(columns=4).classes("w-full gap-4"):
                for ticker in current.profile.watchlist[:8]:
                    snapshot = current.market.get(ticker)
                    if snapshot is None:
                        continue
                    with ui.card().classes("glass-card"):
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label(ticker).classes("text-lg font-bold")
                            ui.label(snapshot.source).classes("neon-pill")
                        ui.label(_money(snapshot.last)).classes("text-2xl font-bold")
                        ui.label(_signed_percent(snapshot.change_percent)).classes(_color_class(snapshot.change_percent))
                        ui.label(
                            f"RSI {snapshot.technicals.rsi:.1f} | Vol spike {snapshot.technicals.volume_spike:.2f}x | Trend {snapshot.technicals.trend}"
                        ).classes("subtle")

        @ui.refreshable
        def chart_panel() -> None:
            current = platform.get_state()
            watchlist = current.profile.watchlist or settings.watchlist
            if selected_ticker["value"] not in watchlist:
                selected_ticker["value"] = watchlist[0]
            snapshot = current.market.get(selected_ticker["value"])
            with ui.row().classes("w-full gap-4 no-wrap max-[1100px]:flex-wrap"):
                with ui.card().classes("glass-card w-full"):
                    ui.label("Live Market Engine").classes("title-font text-xl")
                    ui.select(
                        options=watchlist,
                        value=selected_ticker["value"],
                        on_change=lambda event: (selected_ticker.__setitem__("value", event.value), chart_panel.refresh()),
                    ).classes("w-48")
                    if snapshot is not None:
                        ui.plotly(_price_figure(snapshot)).classes("w-full")
                with ui.card().classes("glass-card min-w-[320px]"):
                    ui.label("Ticker Intelligence").classes("title-font text-xl")
                    if snapshot is not None:
                        with ui.column().classes("w-full gap-3"):
                            for label, value in [
                                ("Prediction", f"{snapshot.prediction.label} ({snapshot.prediction.probability_up:.0%} up)"),
                                ("Expected Move", _percent(snapshot.prediction.expected_move_pct)),
                                ("Spread", _money(snapshot.spread or 0)),
                                ("Volatility", f"{snapshot.volatility:.2f}"),
                                ("Patterns", ", ".join(snapshot.technicals.patterns or ['none'])),
                            ]:
                                with ui.row().classes("w-full items-center justify-between ai-badge"):
                                    ui.label(label).classes("subtle")
                                    ui.label(value).classes("font-semibold")

        @ui.refreshable
        def ai_panel() -> None:
            current = platform.get_state()
            with ui.row().classes("w-full gap-4 no-wrap max-[1100px]:flex-wrap"):
                with ui.card().classes("glass-card w-full"):
                    ui.label("AI Brain").classes("title-font text-xl")
                    ui.label(current.brain.summary).classes("subtle")
                    ui.linear_progress(value=current.brain.confidence, color="cyan").classes("mt-2")
                    ui.label(f"Confidence {current.brain.confidence:.0%} | Sentiment score {current.brain.market_sentiment_score:.2f}").classes("subtle")
                    ui.separator()
                    ui.label("Autonomous insights").classes("font-semibold")
                    for insight in current.brain.opportunities[:3]:
                        with ui.column().classes("insight-item"):
                            ui.label(f"{insight.ticker}: {insight.headline}").classes("font-semibold")
                            ui.label(insight.rationale).classes("subtle")
                            ui.label(f"Action: {insight.action} | Risk: {insight.risk_level} | Confidence: {insight.confidence:.0%}").classes("subtle")
                with ui.card().classes("glass-card w-full"):
                    ui.label("Risk warnings").classes("title-font text-xl")
                    for warning in current.brain.warnings[:3]:
                        with ui.column().classes("insight-item"):
                            ui.label(f"{warning.ticker}: {warning.headline}").classes("font-semibold negative")
                            ui.label(warning.rationale).classes("subtle")
                    ui.separator()
                    ui.label("AI-generated strategy").classes("font-semibold")
                    for note in current.brain.strategy_notes:
                        ui.label(f"• {note}").classes("subtle")

        @ui.refreshable
        def portfolio_panel() -> None:
            current = platform.get_state()
            with ui.row().classes("w-full gap-4 no-wrap max-[1100px]:flex-wrap"):
                with ui.card().classes("glass-card w-full"):
                    ui.label("Portfolio Tracker").classes("title-font text-xl")
                    ui.label(
                        f"Cash {_money(current.portfolio.cash)} | Total PnL {_signed_money(current.portfolio.total_pnl)} | Expenses this month {_money(current.portfolio.expenses_month)}"
                    ).classes("subtle")
                    if current.portfolio.positions:
                        for position in current.portfolio.positions[:8]:
                            with ui.row().classes("w-full items-center justify-between ai-badge"):
                                ui.label(f"{position.ticker} {position.shares:.2f} sh")
                                ui.label(
                                    f"{_money(position.market_value)} | {_signed_money(position.pnl_unrealized)} | alloc {position.allocation:.0%}"
                                ).classes(_color_class(position.pnl_unrealized))
                    else:
                        ui.label("No live positions stored yet. Add manual positions below or wire an optional connector.").classes("subtle")
                    ui.separator()
                    for advice in current.portfolio.budget_advice:
                        ui.label(f"• {advice}").classes("subtle")
                with ui.card().classes("glass-card w-full"):
                    ui.plotly(_risk_heatmap_figure(current)).classes("w-full")

        @ui.refreshable
        def news_panel() -> None:
            current = platform.get_state()
            with ui.row().classes("w-full gap-4 no-wrap max-[1100px]:flex-wrap"):
                with ui.card().classes("glass-card w-full"):
                    ui.label("News + Sentiment Engine").classes("title-font text-xl")
                    for ticker, items in list(current.news.items())[:4]:
                        ui.label(ticker).classes("font-semibold mt-2")
                        for item in items[:2]:
                            with ui.column().classes("news-item mb-2"):
                                with ui.row().classes("w-full items-center justify-between"):
                                    ui.link(item.title, item.url, new_tab=True).classes("font-semibold")
                                    ui.label(item.sentiment).classes(_color_class(item.sentiment_score))
                                ui.label(
                                    f"{item.source} | credibility {item.credibility:.0%} | relevance {item.relevance:.0%}"
                                ).classes("subtle")
                                if item.flags:
                                    ui.label("Flags: " + ", ".join(item.flags)).classes("subtle")

        @ui.refreshable
        def simulator_panel() -> None:
            current = platform.get_state()
            with ui.row().classes("w-full gap-4 no-wrap max-[1100px]:flex-wrap"):
                with ui.card().classes("glass-card w-full"):
                    ui.label("Paper Trading Simulator").classes("title-font text-xl")
                    ui.label(
                        f"Equity {_signed_money(current.simulation.equity)} | Gross exposure {_money(current.simulation.gross_exposure)}"
                    ).classes("subtle")
                    ui.plotly(_simulation_figure(current)).classes("w-full")
                with ui.card().classes("glass-card w-full"):
                    ui.label("What-if Scenario Engine").classes("title-font text-xl")
                    ui.label(current.scenario.narrative if current.scenario else "Run a stress test").classes("subtle")
                    if current.scenario:
                        for impact in current.scenario.impacts[:5]:
                            with ui.row().classes("w-full items-center justify-between ai-badge"):
                                ui.label(f"{impact.ticker} shock {_signed_percent(impact.shock_pct)}")
                                ui.label(_signed_money(impact.estimated_pnl)).classes(_color_class(impact.estimated_pnl))

        hero_metrics()
        market_strip()
        chart_panel()
        ai_panel()
        portfolio_panel()
        news_panel()
        simulator_panel()

        with ui.row().classes("w-full gap-4 no-wrap max-[1100px]:flex-wrap"):
            with ui.card().classes("glass-card w-full"):
                ui.label("Profile & Goal Controls").classes("title-font text-xl")
                with ui.row().classes("w-full gap-3"):
                    horizon_select = ui.select(
                        options=["short-term", "balanced", "long-term"],
                        value=state.profile.investment_horizon,
                        label="Horizon",
                    ).classes("w-48")
                    risk_select = ui.select(
                        options=["conservative", "moderate", "aggressive"],
                        value=state.profile.risk_tolerance,
                        label="Risk",
                    ).classes("w-48")
                    budget_input = ui.number("Monthly budget", value=state.profile.monthly_budget).classes("w-40")
                    cash_input = ui.number("Cash", value=state.profile.cash_balance).classes("w-40")
                watchlist_input = ui.input("Watchlist CSV", value=", ".join(state.profile.watchlist)).classes("w-full")
                ui.button("Save profile", on_click=save_profile)

            with ui.card().classes("glass-card w-full"):
                ui.label("Manual Portfolio Input").classes("title-font text-xl")
                with ui.row().classes("w-full gap-3"):
                    position_ticker = ui.input("Ticker", value="AAPL").classes("w-24")
                    position_shares = ui.number("Shares", value=10).classes("w-28")
                    position_cost = ui.number("Avg cost", value=180).classes("w-32")
                    position_thesis = ui.input("Thesis", value="Core quality long").classes("grow")
                ui.button("Save position", on_click=save_position)
                ui.separator()
                ui.label("Expense Logger").classes("font-semibold")
                with ui.row().classes("w-full gap-3"):
                    expense_category = ui.input("Category", value="Dining").classes("w-40")
                    expense_amount = ui.number("Amount", value=45).classes("w-28")
                    expense_merchant = ui.input("Merchant", value="Sample merchant").classes("grow")
                ui.button("Log expense", on_click=save_expense)

            with ui.card().classes("glass-card w-full"):
                ui.label("Paper Trade Controls").classes("title-font text-xl")
                with ui.row().classes("w-full gap-3"):
                    sim_ticker = ui.input("Ticker", value=selected_ticker["value"]).classes("w-24")
                    sim_shares = ui.number("Shares", value=5).classes("w-28")
                with ui.row().classes("gap-3"):
                    ui.button("Sim BUY", on_click=lambda: asyncio.create_task(execute_trade("BUY")))
                    ui.button("Sim SELL", on_click=lambda: asyncio.create_task(execute_trade("SELL")))
                ui.separator()
                ui.label("Scenario shock").classes("font-semibold")
                scenario_slider = ui.slider(min=-15, max=15, value=-5, step=1).classes("w-full")
                ui.button("Run scenario", on_click=update_scenario)

    chat_container = ui.card().classes("chat-card fixed bottom-4 right-4 z-50")
    with chat_container:
        ui.label("AI Assistant").classes("title-font text-xl")
        ui.label("Context-aware portfolio and market copilot").classes("subtle")

        @ui.refreshable
        def chat_history() -> None:
            for message in platform.store.list_recent_chat_messages(settings.default_user_id, limit=10):
                css = "chat-bubble"
                if message.role == "assistant":
                    css += " border-cyan-400"
                with ui.column().classes(css):
                    ui.label(message.role.upper()).classes("metric-title text-xs")
                    ui.label(message.message).classes("subtle")
                    ui.label(message.timestamp.strftime("%H:%M:%S")).classes("metric-title text-xs")

        chat_history()
        typing_note = ui.label("AI is reasoning...").classes("typing subtle")
        typing_note.set_visibility(False)
        chat_input = ui.input("Ask about goals, tickers, risk, or strategy").classes("w-full")
        with ui.row().classes("w-full gap-2"):
            ui.button("Send", on_click=lambda: asyncio.create_task(send_chat()))
        ui.label("Suggested prompts").classes("font-semibold mt-2")
        for suggestion in platform.chat_service.suggested_prompts():
            ui.button(suggestion, on_click=lambda _, prompt=suggestion: asyncio.create_task(send_chat(prompt))).props("flat").classes("w-full")

    ui.timer(settings.refresh_seconds, lambda: asyncio.create_task(timed_refresh()))


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title=settings.app_name,
        reload=False,
        favicon="💹",
        host="0.0.0.0",
        port=8080,
    )
