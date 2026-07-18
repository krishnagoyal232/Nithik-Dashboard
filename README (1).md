# Nifty Sector Stock Screener

A single-file Streamlit app. Type a company name, get: P/E vs sector
median, PEG or top-line/bottom-line trend screen, ratio ranking across
the sector's surviving stocks, and trend-based forecasts vs the stock's
own history, its sector, and the Nifty 50.

**Data-driven screening tool, not investment advice.** Every "forecast"
is a naive extrapolation of past price trend — nothing reliably predicts
future returns, and the app makes no such claim.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501` in your browser.

## Put it on your phone too (free, no install for end users)

Streamlit apps deployed to Streamlit Community Cloud are just a normal
web page — the same URL opens fine in a phone browser or a desktop
browser, nothing to install.

1. Push this repo to GitHub (just `streamlit_app.py` and
   `requirements.txt` are needed).
2. Go to https://share.streamlit.io → "New app" → sign in with GitHub →
   pick this repo and `streamlit_app.py`.
3. It builds and gives you a URL like `https://your-app.streamlit.app`.
   Open that on your phone — it's a normal website, works the same as
   on desktop.

## Requires internet access

The app scrapes screener.in and moneycontrol.com live, and pulls price
history from Yahoo Finance — it needs real internet access to work
(Streamlit Cloud has this; your own machine needs to too).

## What changed in v2 (fixes from the first deployed run)

- **Price-history crash fixed.** v1 guessed a Yahoo Finance ticker from
  the company's *display name* (e.g. "LODHA DEVELOPERS LTD.NS", which
  isn't a real ticker → crash). v2 instead reads the real NSE trading
  symbol that screener.in already embeds in its own company URLs (e.g.
  `/company/LODHA/...` → `LODHA`), for both the main company and every
  sector peer. Falls back to a `.BO` (BSE) try if `.NS` has no data.
- **Ratio table showing "None" fixed.** Debt/Equity, Interest Coverage,
  ROE and Net Profit Margin are now computed directly from screener.in's
  own P&L and balance-sheet data — reliable, already scraped, no second
  site needed. moneycontrol is now only a best-effort bonus for
  Current/Quick Ratio (which genuinely can't be derived from screener's
  free tier); if that fails, the app says so plainly instead of showing
  blank cells, and drops those two columns from the ranking rather than
  ranking on mostly-missing data.
- **Charts added**: revenue/net-profit trend, price history + trend
  forecast, and a CAGR comparison bar chart — all shown directly on the
  page.
- **Simpler UI**: a plain-English ✅/❌ verdict card up top, charts shown
  immediately, and the detailed step-by-step working (sector sweep,
  ratio table) tucked into an expander so a non-technical user isn't
  faced with a wall of text by default.

## Notes / limitations

- **Scraping is still inherently fragile** — screener.in and
  moneycontrol have no public API; this uses the same endpoints their
  own sites use. If either changes its HTML, the relevant parsing
  function may need a small update. This is normal maintenance for
  scraping-based tools, not something a config change can fully prevent.
- **Current Ratio / Quick Ratio** are only available on a best-effort
  basis via moneycontrol — free sources generally don't publish the
  current-assets/liabilities split needed to compute them otherwise.
- **PEG's growth-rate input** uses screener's "Compounded Profit Growth"
  (longest available window: 10y → 5y → 3y → TTM).
- **Sector benchmark** falls back to an equal-weighted peer index if no
  clean Yahoo Finance ticker exists for that sector's official index.
- **Forecasts** are log-linear trend extrapolation with a
  volatility-based band — illustrative only, never investment advice.
