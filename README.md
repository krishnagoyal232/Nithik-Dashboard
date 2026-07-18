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

## Notes / limitations

- **Scraping is fragile by nature** — screener.in and moneycontrol have
  no public API; this uses the same endpoints their own sites use. If
  either changes its HTML, the relevant parsing function may need a
  small update.
- **Current Ratio / Quick Ratio** can only come from moneycontrol —
  screener.in's free tier doesn't publish the current-assets/liabilities
  split needed to derive them, so they show as missing if moneycontrol
  scraping fails for a stock.
- **PEG's growth-rate input** uses screener's "Compounded Profit Growth"
  (longest available window: 10y → 5y → 3y → TTM) as the standard proxy.
- **Sector benchmark** falls back to an equal-weighted peer index if no
  clean Yahoo Finance ticker exists for that sector's official index.
- **Forecasts** are log-linear trend extrapolation with a
  volatility-based band — illustrative only, never investment advice.
