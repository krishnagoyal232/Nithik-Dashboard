"""
Nifty Sector Stock Screener — single-file Streamlit app.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Deploy free (so it opens as a normal web page/app on phone AND desktop,
no install needed for whoever uses it):
    1. Push this repo to GitHub.
    2. Go to https://share.streamlit.io, sign in with GitHub.
    3. Point it at this repo / streamlit_app.py. It gives you a URL
       (e.g. https://your-app.streamlit.app) that opens like any website
       on any phone or desktop browser.

IMPORTANT CAVEAT: every "forecast" in this app is a naive extrapolation
of *past* price trend — nothing reliably predicts future stock returns,
and this app makes no such claim. Treat forward-looking numbers as
"if the recent trend mechanically continued," not as predictions, and
never as investment advice. This is a data-driven screening tool.
"""

import re
import time
import statistics
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None


# =============================================================================
# CONFIG — thresholds pulled from the uploaded fundamental-analysis file,
# tune here rather than hunting through the logic below.
# =============================================================================
PEG_THRESHOLD = 1.5                 # PEG >= this -> reject (confirmed against the source file)
REJECT_NEGATIVE_PEG = True

STABLE_FLUCTUATION_BAND = 0.125     # +-10-15% -> "stable/constant growth"
EXCEPTIONAL_SWING_THRESHOLD = 0.25  # >25% single-year swing -> ignore as exceptional
COVID_EXCEPTION_YEARS = {2020, 2021}
MAX_NEGATIVE_PAT_YEARS = 3
RECENT_WINDOW_YEARS = 5
LONG_WINDOW_YEARS = 10

RATIO_DEFINITIONS = {
    "Current Ratio": "higher",
    "Quick Ratio": "higher",
    "Debt/Equity Ratio": "lower",
    "Interest Coverage Ratio": "higher",
    "Return on Equity Ratio": "higher",
    "Net Profit Margin Ratio": "higher",
}

PRICE_HISTORY_YEARS = 10
MIN_PRICE_HISTORY_YEARS = 5
FORECAST_HORIZON_YEARS = 3
NIFTY50_TICKER = "^NSEI"

SECTOR_INDEX_TICKERS = {
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY ENERGY": "^CNXENERGY",
}

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
REQUEST_TIMEOUT = 15
REQUEST_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


# =============================================================================
# HTTP HELPER
# =============================================================================
class FetchError(Exception):
    pass


def http_get(url, params=None, as_json=False):
    last_exc = None
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json() if as_json else resp.text
            if resp.status_code == 404:
                raise FetchError(f"404 Not Found: {url}")
            last_exc = FetchError(f"HTTP {resp.status_code} for {url}")
        except requests.RequestException as e:
            last_exc = e
        if attempt < REQUEST_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise FetchError(f"Failed to fetch {url} after {REQUEST_RETRIES} attempts: {last_exc}")


# =============================================================================
# SCREENER.IN SCRAPER — P/E, peer P/E table, Sales/Net Profit history,
# compounded growth (used for PEG). No official API; uses the same search
# endpoint and HTML the site itself renders. Selectors have fallbacks but
# may need small tweaks if screener changes markup (see README).
# =============================================================================
SCREENER_BASE = "https://www.screener.in"
SCREENER_SEARCH_API = f"{SCREENER_BASE}/api/company/search/"


def _num(text):
    if text is None:
        return None
    cleaned = text.replace(",", "").strip()
    m = re.search(r"-?\d+\.?\d*", cleaned)
    return float(m.group()) if m else None


def screener_search(name):
    raw = http_get(SCREENER_SEARCH_API, params={"q": name}, as_json=True)
    if not raw:
        return []
    return [{"id": i.get("id"), "name": i.get("name"), "url": i.get("url") or f"/company/{i.get('id')}/"} for i in raw]


def _best_match(name, candidates):
    if not candidates:
        return None
    name_low = name.strip().lower()
    for c in candidates:
        if c["name"].strip().lower() == name_low:
            return c
    for c in candidates:
        if name_low in c["name"].strip().lower():
            return c
    return candidates[0]


def screener_fetch_company_page(url_path):
    path = url_path.rstrip("/") + "/"
    for u in (f"{SCREENER_BASE}{path}consolidated/", f"{SCREENER_BASE}{path}"):
        try:
            return BeautifulSoup(http_get(u), "lxml"), u
        except FetchError:
            continue
    raise FetchError(f"Could not fetch screener.in page for {url_path}")


def screener_parse_overview(soup):
    out = {}
    container = soup.find(id="top-ratios") or soup.find("ul", class_=re.compile("ratios|company-ratios"))
    items = container.find_all("li") if container else soup.find_all("li")
    for li in items:
        name_el = li.find(class_=re.compile("name"))
        value_el = li.find(class_=re.compile("value|number"))
        if not name_el:
            continue
        label = name_el.get_text(strip=True)
        value_text = value_el.get_text(" ", strip=True) if value_el else li.get_text(" ", strip=True)
        val = _num(value_text)
        if label and val is not None:
            out[label] = val
    return out


def screener_get_stock_pe(overview):
    for key in ("Stock P/E", "P/E", "Price to Earning"):
        if key in overview:
            return overview[key]
    return None


def screener_parse_peers(soup):
    section = soup.find(id="peers") or soup.find(id="peers-table")
    table = section.find("table") if section else soup.find("table", class_=re.compile("data-table"))
    if not table:
        return []
    headers = [th.get_text(strip=True) for th in table.find("thead").find_all("th")] if table.find("thead") else []
    pe_idx, name_idx = None, None
    for i, h in enumerate(headers):
        h_low = h.lower()
        if "p/e" in h_low or h_low == "pe":
            pe_idx = i
        if "name" in h_low:
            name_idx = i
    name_idx = 1 if name_idx is None else name_idx
    pe_idx = 3 if pe_idx is None else pe_idx
    peers = []
    body = table.find("tbody") or table
    for tr in body.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) <= max(name_idx, pe_idx):
            continue
        name = tds[name_idx].get_text(strip=True)
        pe = _num(tds[pe_idx].get_text(strip=True))
        if name:
            peers.append({"name": name, "pe": pe})
    return peers


def _parse_yearly_row(table, row_label_variants):
    if table is None:
        return {}
    thead = table.find("thead")
    year_labels = [th.get_text(strip=True) for th in thead.find_all("th")] if thead else []
    body = table.find("tbody") or table
    for tr in body.find_all("tr"):
        cells = tr.find_all("td") or tr.find_all("th")
        if not cells:
            continue
        label = cells[0].get_text(strip=True).replace("+", "").strip()
        if any(v.lower() == label.lower() for v in row_label_variants):
            values = {}
            for i, td in enumerate(cells[1:], start=1):
                year = year_labels[i] if i < len(year_labels) else f"col{i}"
                values[year] = _num(td.get_text(strip=True))
            return values
    return {}


def screener_parse_profit_loss(soup):
    section = soup.find(id="profit-loss")
    table = section.find("table") if section else None
    sales = _parse_yearly_row(table, ["Sales", "Revenue", "Sales +"])
    net_profit = _parse_yearly_row(table, ["Net Profit", "Net Profit +"])
    return sales, net_profit


def screener_parse_compounded_growth(soup):
    result = {}
    for heading_text, key in (("Compounded Sales Growth", "Sales"), ("Compounded Profit Growth", "Profit")):
        heading = soup.find(string=re.compile(re.escape(heading_text), re.I))
        if not heading:
            continue
        container = heading.find_parent(["table", "div"])
        table = container.find("table") if container and container.name != "table" else container
        if not table:
            continue
        rows = {}
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) >= 2:
                period = tds[0].get_text(strip=True).replace(":", "")
                val = _num(tds[1].get_text(strip=True))
                if period and val is not None:
                    rows[period] = val
        if rows:
            result[key] = rows
    return result


def get_screener_snapshot(name):
    candidates = screener_search(name)
    match = _best_match(name, candidates)
    if not match:
        raise ValueError(f"No screener.in match found for '{name}'")
    soup, used_url = screener_fetch_company_page(match["url"])
    overview = screener_parse_overview(soup)
    stock_pe = screener_get_stock_pe(overview)
    peers = screener_parse_peers(soup)
    sales, net_profit = screener_parse_profit_loss(soup)
    growth = screener_parse_compounded_growth(soup)
    return {
        "name": match["name"], "url": used_url, "overview": overview, "stock_pe": stock_pe,
        "peers": peers, "sales_by_year": sales, "net_profit_by_year": net_profit, "compounded_growth": growth,
    }


# =============================================================================
# MONEYCONTROL SCRAPER — the 6 ratios (Current, Quick, D/E, Interest
# Coverage, ROE, NPM). Most fragile part of the pipeline; falls back to
# partial data derived from screener.in if it fails (see get_ratios below).
# =============================================================================
MC_AUTOSUGGEST_URL = "https://www.moneycontrol.com/mccode/common/autosuggesion.php"

MC_RATIO_LABEL_MAP = {
    "Current Ratio": ["Current Ratio"],
    "Quick Ratio": ["Quick Ratio"],
    "Debt/Equity Ratio": ["Debt Equity Ratio", "Debt/Equity Ratio", "Total Debt/Equity"],
    "Interest Coverage Ratio": ["Interest Cover Ratio", "Interest Coverage Ratio"],
    "Return on Equity Ratio": ["Return On Equity", "Return on Equity"],
    "Net Profit Margin Ratio": ["Net Profit Margin", "Net Profit Margin(%)"],
}


def mc_search(name):
    raw = http_get(MC_AUTOSUGGEST_URL, params={"query": name, "type": "1", "format": "json"}, as_json=True)
    items = raw if isinstance(raw, list) else raw.get("data", []) if isinstance(raw, dict) else []
    out = []
    for item in items:
        link = item.get("link_src") or item.get("url")
        stock_name = item.get("stock_name") or item.get("name")
        if link and stock_name:
            out.append({"name": stock_name, "url": link})
    return out


def _mc_ratios_url(overview_url):
    m = re.search(r"stockpricequote/[^/]+/([^/]+)/([A-Za-z0-9]+)$", overview_url)
    if not m:
        return None
    slug, code = m.group(1), m.group(2)
    return f"https://www.moneycontrol.com/financials/{slug}/ratiosVI/{code}"


def _mc_parse_ratios_page(html):
    soup = BeautifulSoup(html, "lxml")
    out = {}
    table = soup.find("table", class_=re.compile("mctable|table")) or soup.find("table")
    if not table:
        return out
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        label = cells[0].get_text(strip=True)
        value = None
        for td in cells[1:]:
            m = re.search(r"-?\d+\.?\d*", td.get_text(strip=True).replace(",", ""))
            if m:
                value = float(m.group())
                break
        if label and value is not None:
            out[label] = value
    return out


def get_moneycontrol_ratios(name):
    candidates = mc_search(name)
    match = _best_match(name, candidates)
    if not match:
        raise ValueError(f"No moneycontrol match for '{name}'")
    ratios_url = _mc_ratios_url(match["url"])
    if not ratios_url:
        raise ValueError(f"Could not derive ratios URL from {match['url']}")
    raw = _mc_parse_ratios_page(http_get(ratios_url))
    result = {}
    for wanted, variants in MC_RATIO_LABEL_MAP.items():
        for v in variants:
            if v in raw:
                result[wanted] = raw[v]
                break
    return result


def get_ratios_with_fallback(name, snapshot=None):
    """moneycontrol first (has all 6); fall back to partial ratios derived
    from screener.in's overview block if moneycontrol scraping fails.
    Current/Quick ratio genuinely can't be derived from screener's free
    tier (no current-assets/liabilities breakdown), so they stay missing
    in that fallback rather than being guessed."""
    try:
        r = get_moneycontrol_ratios(name)
        if r:
            return r, "moneycontrol"
    except Exception:
        pass
    try:
        snap = snapshot or get_screener_snapshot(name)
        ov = snap["overview"]
        partial = {}
        if "ROE" in ov:
            partial["Return on Equity Ratio"] = ov["ROE"]
        for key in ("Debt to equity", "Debt to Equity"):
            if key in ov:
                partial["Debt/Equity Ratio"] = ov[key]
                break
        return partial, "screener.in (partial — current/quick ratio unavailable)"
    except Exception:
        return {}, "unavailable"


# =============================================================================
# STEP 2 & 3 — median-P/E valuation classification + PEG rule
# =============================================================================
@dataclass
class ValuationResult:
    company_pe: float
    sector_median_pe: float
    bucket: str
    peers_used: int
    peers_dropped_negative_pe: int


def sector_median_pe(company_name, company_pe, peers):
    pe_values, dropped, seen = [], 0, False
    for p in peers:
        if p["name"].strip().lower() == company_name.strip().lower():
            seen = True
        pe = p.get("pe")
        if pe is None:
            continue
        if pe < 0:
            dropped += 1
            continue
        pe_values.append(pe)
    if not seen and company_pe is not None and company_pe >= 0:
        pe_values.append(company_pe)
    if not pe_values:
        raise ValueError("No usable (non-negative) P/E values found among peers")
    return statistics.median(pe_values), len(pe_values), dropped


def classify_valuation(company_name, company_pe, peers) -> ValuationResult:
    if company_pe is None:
        raise ValueError("Company P/E is missing")
    if company_pe < 0:
        raise ValueError(f"{company_name} has a negative P/E ({company_pe}) — excluded per the file's rule")
    median_pe, used, dropped = sector_median_pe(company_name, company_pe, peers)
    bucket = "overvalued" if company_pe >= median_pe else "undervalued"
    return ValuationResult(company_pe, median_pe, bucket, used, dropped)


@dataclass
class PegResult:
    peg: Optional[float]
    growth_rate_used_pct: Optional[float]
    growth_period_used: Optional[str]
    decision: str
    reason: str


def compute_peg(pe, compounded_profit_growth: dict):
    if pe is None:
        return PegResult(None, None, None, "Reject", "No P/E available")
    for window in ("10Years", "5Years", "3Years", "TTM"):
        growth = compounded_profit_growth.get(window)
        if growth is not None and growth != 0:
            peg = pe / growth
            if peg < 0 and REJECT_NEGATIVE_PEG:
                return PegResult(peg, growth, window, "Reject", f"PEG is negative ({peg:.2f})")
            if peg >= PEG_THRESHOLD:
                return PegResult(peg, growth, window, "Reject", f"PEG {peg:.2f} >= threshold {PEG_THRESHOLD}")
            return PegResult(peg, growth, window, "Accept", f"PEG {peg:.2f} < threshold {PEG_THRESHOLD}")
    return PegResult(None, None, None, "Reject", "No usable profit-growth figure to compute PEG")


# =============================================================================
# STEP 4 — top-line/bottom-line trend analysis (undervalued path). 10 rules
# from the uploaded file (rule 10 is a filler line in the source file
# itself and adds no real logic).
# =============================================================================
def _extract_year(label: str):
    m = re.search(r"(20\d{2}|19\d{2})", label)
    if m:
        return int(m.group(1))
    m2 = re.search(r"FY\s*'?(\d{2})", label, re.I)
    return 2000 + int(m2.group(1)) if m2 else None


def _sorted_series(values_by_year):
    out = []
    for label, val in values_by_year.items():
        if val is None or label.strip().upper() == "TTM":
            continue
        year = _extract_year(label)
        if year is not None:
            out.append((year, val))
    out.sort(key=lambda t: t[0])
    return out


def _yoy_growth(series):
    growth = []
    for (y0, v0), (y1, v1) in zip(series, series[1:]):
        if v0 == 0:
            continue
        growth.append((y1, (v1 - v0) / abs(v0)))
    return growth


@dataclass
class LineTrend:
    direction: str
    filtered_growth: List[Tuple[int, float]]
    negative_years: int = 0
    negative_in_recent_window: int = 0
    notes: List[str] = field(default_factory=list)


def _filter_exceptions(growth, notes):
    big_swing_years = [y for y, g in growth if abs(g) > EXCEPTIONAL_SWING_THRESHOLD and y not in COVID_EXCEPTION_YEARS]
    recurring = len(big_swing_years) > 1
    filtered = []
    for y, g in growth:
        if y in COVID_EXCEPTION_YEARS:
            notes.append(f"{y}: excluded as COVID-year exception (rule 7)")
            continue
        if abs(g) > EXCEPTIONAL_SWING_THRESHOLD and not recurring:
            notes.append(f"{y}: {g:+.0%} swing treated as one-off exceptional year (rule 5), excluded")
            continue
        filtered.append((y, g))
    if recurring:
        notes.append(f"Large (>25%) swings recurred in {len(big_swing_years)} non-COVID years — treated as real volatility (rule 5 proviso)")
    return filtered


def _classify_line(values_by_year, is_profit_line):
    series = _sorted_series(values_by_year)
    notes = []
    negative_years = sum(1 for _, v in series if v < 0) if is_profit_line else 0
    recent_cutoff = max((y for y, _ in series), default=0) - RECENT_WINDOW_YEARS + 1
    negative_recent = sum(1 for y, v in series if is_profit_line and v < 0 and y >= recent_cutoff)

    if len(series) < 2:
        return LineTrend("insufficient_data", [], negative_years, negative_recent, ["Not enough yearly data points"])

    growth = _yoy_growth(series)
    filtered = _filter_exceptions(growth, notes)
    if not filtered:
        return LineTrend("insufficient_data", [], negative_years, negative_recent, notes + ["All years filtered as exceptions"])

    within_band = all(abs(g) <= STABLE_FLUCTUATION_BAND for _, g in filtered)
    has_up = any(g > 0 for _, g in filtered)
    has_down = any(g < 0 for _, g in filtered)
    if within_band and has_up and has_down:
        notes.append(f"All {len(filtered)} usable years within +-{STABLE_FLUCTUATION_BAND:.0%} band and mixed sign -> stable/constant growth (rule 4)")
        return LineTrend("stable", filtered, negative_years, negative_recent, notes)

    positive_years = sum(1 for _, g in filtered if g > 0)
    direction = "increasing" if positive_years >= len(filtered) / 2 else "decreasing"
    notes.append(f"{positive_years}/{len(filtered)} usable years positive -> classified '{direction}'")
    return LineTrend(direction, filtered, negative_years, negative_recent, notes)


@dataclass
class TrendResult:
    decision: str
    top_line: LineTrend
    bottom_line: LineTrend
    window_used_years: int
    reason: str
    all_notes: List[str] = field(default_factory=list)


def _decide_trend(top, bottom):
    if bottom.negative_years > MAX_NEGATIVE_PAT_YEARS:
        if bottom.negative_years and bottom.negative_in_recent_window / bottom.negative_years > 0.5:
            return "Reject", f"Bottom line had {bottom.negative_years} loss-making years, over half within the last {RECENT_WINDOW_YEARS} years (rule 6)"
    if top.direction == "insufficient_data" or bottom.direction == "insufficient_data":
        return None, "Inconclusive with available data"
    if top.direction == "decreasing" and bottom.direction == "decreasing":
        return "Reject", "Both top and bottom line in downtrend (rule 9)"
    if top.direction in ("increasing", "stable") and bottom.direction in ("increasing", "stable"):
        return "Accept", "Both top and bottom line increasing/stable (rule 1 / rule 4)"
    if top.direction == "increasing" and bottom.direction == "decreasing":
        return "Accept", "Top line increasing, bottom line decreasing (rule 2)"
    if bottom.direction == "increasing" and top.direction == "decreasing":
        return "Accept", "Bottom line increasing, top line decreasing (rule 3)"
    if top.direction == "stable" or bottom.direction == "stable":
        return "Accept", "One line stable/constant growth, other not clearly down (rule 4)"
    return "Reject", "Did not match any acceptance rule"


def analyze_trend(sales_by_year, net_profit_by_year) -> TrendResult:
    top = _classify_line(sales_by_year, False)
    bottom = _classify_line(net_profit_by_year, True)
    decision, reason = _decide_trend(top, bottom)
    notes = list(top.notes) + list(bottom.notes)

    if decision is None:
        notes.append(f"10-year analysis inconclusive -> falling back to last {RECENT_WINDOW_YEARS} years (rule 8)")

        def _last_n(values_by_year, n):
            series = _sorted_series(values_by_year)
            cutoff_years = {y for y, _ in series[-n:]}
            return {lbl: v for lbl, v in values_by_year.items() if _extract_year(lbl) in cutoff_years}

        top5 = _classify_line(_last_n(sales_by_year, RECENT_WINDOW_YEARS), False)
        bottom5 = _classify_line(_last_n(net_profit_by_year, RECENT_WINDOW_YEARS), True)
        decision, reason = _decide_trend(top5, bottom5)
        if decision is None:
            decision, reason = "Reject", "Insufficient data even in the last 5 years"
        notes += top5.notes + bottom5.notes
        return TrendResult(decision, top5, bottom5, RECENT_WINDOW_YEARS, reason, notes)

    return TrendResult(decision, top, bottom, LONG_WINDOW_YEARS, reason, notes)


# =============================================================================
# STEP 5 — ratio ranking across selected sector stocks
# =============================================================================
def rank_companies(ratios_by_company: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(ratios_by_company, orient="index")
    df = df.reindex(columns=list(RATIO_DEFINITIONS.keys()))
    rank_df = pd.DataFrame(index=df.index)
    for ratio, direction in RATIO_DEFINITIONS.items():
        rank_df[ratio + " Rank"] = df[ratio].rank(ascending=(direction == "lower"), method="average")
    rank_df["Sum Of Ranks"] = rank_df.sum(axis=1, skipna=True)
    rank_df["Ratios Missing"] = df.isna().sum(axis=1)
    rank_df["Final Rank"] = rank_df["Sum Of Ranks"].rank(ascending=True, method="min").astype(int)
    return pd.concat([df, rank_df], axis=1).sort_values("Final Rank")


# =============================================================================
# STEPS 6-8 — trend-based forecasting & benchmarking (illustrative only)
# =============================================================================
def to_nse_ticker(symbol: str) -> str:
    symbol = symbol.strip().upper()
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"


def get_price_history(symbol, years=PRICE_HISTORY_YEARS) -> pd.Series:
    if yf is None:
        raise RuntimeError("yfinance is not installed — add it via requirements.txt")
    ticker = to_nse_ticker(symbol)
    df = yf.download(ticker, period=f"{years}y", interval="1d", progress=False, auto_adjust=True)
    if df.empty or len(df) < MIN_PRICE_HISTORY_YEARS * 200:
        df = yf.download(ticker, period=f"{MIN_PRICE_HISTORY_YEARS}y", interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price history for {ticker}")
    close = df["Close"].dropna()
    close.name = symbol
    return close


def cagr(series: pd.Series) -> float:
    if len(series) < 2:
        return float("nan")
    years = (series.index[-1] - series.index[0]).days / 365.25
    return float("nan") if years <= 0 else (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1


def _log_linear_trend(series):
    t_years = (series.index - series.index[0]).days / 365.25
    log_p = np.log(series.values)
    b, a = np.polyfit(t_years, log_p, 1)
    fitted = a + b * t_years
    resid_std = np.std(log_p - fitted)
    return b, a, resid_std, t_years


def project_forward(series: pd.Series, horizon_years=FORECAST_HORIZON_YEARS):
    b, a, resid_std, t_years = _log_linear_trend(series)
    last_t = t_years[-1]
    future_t = np.linspace(last_t, last_t + horizon_years, max(horizon_years * 12, 12))
    future_log_p = a + b * future_t
    band = resid_std * np.sqrt(np.maximum(future_t - last_t, 0.01))
    projected_price = float(np.exp(future_log_p[-1]))
    implied_cagr = (projected_price / series.iloc[-1]) ** (1 / horizon_years) - 1
    return {
        "future_t": future_t, "future_price": np.exp(future_log_p),
        "future_price_hi": np.exp(future_log_p + band), "future_price_lo": np.exp(future_log_p - band),
        "projected_price": projected_price, "implied_forward_cagr": implied_cagr,
    }


def get_sector_benchmark_series(sector_index_ticker, peer_symbols, years=PRICE_HISTORY_YEARS):
    if sector_index_ticker and yf is not None:
        try:
            df = yf.download(sector_index_ticker, period=f"{years}y", interval="1d", progress=False, auto_adjust=True)
            if not df.empty:
                s = df["Close"].dropna()
                s.name = "Sector Index"
                return s, "official sectoral index"
        except Exception:
            pass
    frames = []
    for sym in peer_symbols:
        try:
            frames.append(get_price_history(sym, years))
        except Exception:
            continue
    if not frames:
        raise ValueError("Could not build a sector benchmark")
    aligned = pd.concat(frames, axis=1).dropna(how="all")
    normalized = aligned.apply(lambda col: col / col.dropna().iloc[0] * 1000)
    ew = normalized.mean(axis=1)
    ew.name = "Sector Index (equal-weighted peer proxy)"
    return ew, "equal-weighted peer proxy"


def get_nifty50_series(years=PRICE_HISTORY_YEARS):
    df = yf.download(NIFTY50_TICKER, period=f"{years}y", interval="1d", progress=False, auto_adjust=True)
    s = df["Close"].dropna()
    s.name = "Nifty 50"
    return s


def build_cagr_comparison(entries: dict) -> pd.DataFrame:
    rows = []
    for label, series in entries.items():
        try:
            hist = cagr(series)
            proj = project_forward(series)["implied_forward_cagr"]
        except Exception:
            hist, proj = float("nan"), float("nan")
        rows.append({"Entity": label, "Historical CAGR": hist, "Trend-Projected Forward CAGR": proj})
    return pd.DataFrame(rows).set_index("Entity")


def plot_price_with_forecast(series, company_label):
    forecast = project_forward(series)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(series.index, series.values, label=f"{company_label} — historical", color="#1f77b4")
    future_dates = series.index[-1] + pd.to_timedelta((forecast["future_t"] - forecast["future_t"][0]) * 365.25, unit="D")
    ax.plot(future_dates, forecast["future_price"], "--", color="#d62728", label="Naive trend projection")
    ax.fill_between(future_dates, forecast["future_price_lo"], forecast["future_price_hi"], color="#d62728", alpha=0.15, label="Historical-volatility band (not a forecast interval)")
    ax.set_title(f"{company_label}: price history & naive trend projection\n(illustrative only — not a prediction)")
    ax.set_ylabel("Price (Rs.)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_cagr_bars(comparison_df, title):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    comparison_df[["Historical CAGR", "Trend-Projected Forward CAGR"]].plot(kind="bar", ax=ax, color=["#1f77b4", "#d62728"])
    ax.set_ylabel("CAGR")
    ax.set_title(title + "\n(projected figures are illustrative extrapolations, not predictions)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    fig.tight_layout()
    return fig


# =============================================================================
# ORCHESTRATION — steps 1-5 for one company (used both for the main
# company and for sweeping its sector peers)
# =============================================================================
def classify_one_company(name, snapshot=None):
    result = {"name": name, "selected": False, "error": None}
    try:
        snap = snapshot or get_screener_snapshot(name)
        pe = snap["stock_pe"]
        result["pe"] = pe
        result["screener_url"] = snap["url"]
        val = classify_valuation(snap["name"], pe, snap["peers"])
        result["bucket"] = val.bucket
        result["sector_median_pe"] = val.sector_median_pe
        if val.bucket == "overvalued":
            growth = snap["compounded_growth"].get("Profit", {})
            peg = compute_peg(pe, growth)
            result["peg"] = peg.peg
            result["decision"] = peg.decision
            result["reason"] = peg.reason
        else:
            trend = analyze_trend(snap["sales_by_year"], snap["net_profit_by_year"])
            result["decision"] = trend.decision
            result["reason"] = trend.reason
            result["trend_notes"] = trend.all_notes
        result["selected"] = result["decision"] == "Accept"
    except Exception as e:
        result["error"] = str(e)
    return result


# =============================================================================
# STREAMLIT UI — this is what makes it a single "app" that opens the same
# way on a phone browser or a desktop browser once deployed.
# =============================================================================
st.set_page_config(page_title="Nifty Sector Stock Screener", layout="centered")
st.title("📊 Nifty Sector Stock Screener")
st.caption(
    "Data-driven screening tool — not investment advice. All forecasts are "
    "trend extrapolations of past price history, not predictions."
)

company_name = st.text_input("Company name (as listed on screener.in)", placeholder="e.g. DLF, Godrej Properties")
run_clicked = st.button("Run analysis", type="primary")

if run_clicked and company_name.strip():
    try:
        with st.spinner("Step 1: fetching P/E and sector peers from screener.in..."):
            snap = get_screener_snapshot(company_name)
        st.success(f"Found **{snap['name']}** — Stock P/E: {snap['stock_pe']}")
        st.write("Peers:", ", ".join(p["name"] for p in snap["peers"]) or "none found")

        with st.spinner("Step 2-3/4: classifying valuation and applying PEG/trend rule..."):
            main_result = classify_one_company(snap["name"], snapshot=snap)

        if main_result["error"]:
            st.error(main_result["error"])
        else:
            st.subheader("Valuation classification")
            st.write(f"Sector median P/E: **{main_result['sector_median_pe']:.2f}**")
            st.write(f"Bucket: **{main_result['bucket'].upper()}**")
            st.write(f"Decision: **{main_result['decision']}** — {main_result['reason']}")

            st.subheader("Step 5: sector ratio ranking")
            with st.spinner("Sweeping sector peers through the same rules..."):
                sector_results = {snap["name"]: main_result}
                for peer in snap["peers"]:
                    if peer["name"].strip().lower() == snap["name"].strip().lower():
                        continue
                    sector_results[peer["name"]] = classify_one_company(peer["name"])
                    time.sleep(1)
                selected = [n for n, r in sector_results.items() if r.get("selected")]

            st.write("Selected stocks (passed valuation/PEG/trend rules):", ", ".join(selected) or "none")

            ranked_df = None
            with st.spinner("Fetching ratios for selected stocks..."):
                ratio_data = {}
                for c in selected:
                    r, _src = get_ratios_with_fallback(c)
                    if r:
                        ratio_data[c] = r
                    time.sleep(1)
                if ratio_data:
                    ranked_df = rank_companies(ratio_data)
                    st.dataframe(ranked_df)
                else:
                    st.warning("No ratio data could be retrieved for the selected stocks.")

            st.subheader("Steps 6-8: forecasting (illustrative trend extrapolation)")
            with st.spinner("Pulling price history and building forecasts..."):
                own_prices = get_price_history(snap["name"])
                own_cagr = cagr(own_prices)
                own_forecast = project_forward(own_prices)
                beats_own = own_forecast["implied_forward_cagr"] > own_cagr

                st.write(f"Historical CAGR: **{own_cagr:.1%}** | Trend-projected forward CAGR: **{own_forecast['implied_forward_cagr']:.1%}**")
                st.write("✅ Beats its own historical return" if beats_own else "❌ Does not beat its own historical return")
                st.pyplot(plot_price_with_forecast(own_prices, snap["name"]))

                peer_symbols = [p["name"] for p in snap["peers"] if p["name"].strip().lower() != snap["name"].strip().lower()]
                sector_series, sector_source = get_sector_benchmark_series(None, peer_symbols)
                nifty_series = get_nifty50_series()

                comparison_entries = {snap["name"]: own_prices, f"Sector ({sector_source})": sector_series, "Nifty 50": nifty_series}
                if ranked_df is not None:
                    for tp in [c for c in ranked_df.index[:2] if c != snap["name"]]:
                        try:
                            comparison_entries[tp] = get_price_history(tp)
                        except Exception:
                            pass

                comp_df = build_cagr_comparison(comparison_entries)
                st.dataframe(comp_df.style.format("{:.1%}"))
                st.pyplot(plot_cagr_bars(comp_df, f"{snap['name']} vs sector, Nifty 50 & top sector performers"))

        st.info("This is a data-driven screen, not investment advice — do your own further diligence before acting on it.")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.code(traceback.format_exc())

elif run_clicked:
    st.warning("Enter a company name first.")
