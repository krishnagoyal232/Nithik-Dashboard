"""
Nifty Sector Stock Screener — single-file Streamlit app.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Deploy free (opens as a normal web page on phone AND desktop):
    1. Push this repo to GitHub.
    2. https://share.streamlit.io -> New app -> point at this repo/file.
    3. You get a URL like https://your-app.streamlit.app — opens like
       any website, no install needed for whoever uses it.

CAVEAT: every "forecast" here is a trend extrapolation of *past* price
history — not a prediction. This is a data-driven screening tool, not
investment advice.

CHANGELOG (v2 — fixes from the first deployed run):
  - Ticker bug fix: price history now uses the NSE trading symbol
    screener.in itself embeds in its company URLs (e.g. "LODHA"), instead
    of guessing a ticker from the display name ("LODHA DEVELOPERS LTD.NS"
    is not a real ticker, which is why the first version crashed).
  - Ratio bug fix: Debt/Equity, Interest Coverage, ROE and Net Profit
    Margin are now computed directly from screener.in's own P&L/balance
    sheet data (reliable, already-scraped, no extra site needed).
    moneycontrol is now only a best-effort bonus for Current/Quick Ratio,
    which genuinely cannot be derived from screener's free tier — the
    app is upfront in the UI when those two are unavailable instead of
    silently showing blanks.
  - Added a revenue/profit trend chart and kept the price-forecast and
    CAGR-comparison charts.
  - Simplified the UI: a plain-English verdict card up top, charts shown
    directly, and detailed step-by-step working tucked into expanders
    so a non-technical client sees a clean result by default.
"""

import time
import statistics
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import streamlit as st

# requests, bs4, pandas and numpy are core — the app cannot function
# without them (scraping, ratio maths, etc). If any is missing, that
# means requirements.txt wasn't installed for this deployment. Rather
# than let a bare `import` crash with a raw traceback before any UI
# renders (as happened when requirements.txt wasn't picked up), catch
# it here and show one clear, actionable message.
_missing_core = []
try:
    import re
except ImportError:
    _missing_core.append("re")
try:
    import requests
except ImportError:
    _missing_core.append("requests")
try:
    import numpy as np
except ImportError:
    _missing_core.append("numpy")
try:
    import pandas as pd
except ImportError:
    _missing_core.append("pandas")
try:
    from bs4 import BeautifulSoup
except ImportError:
    _missing_core.append("beautifulsoup4")

if _missing_core:
    st.set_page_config(page_title="Nifty Sector Stock Screener — setup needed", layout="centered")
    st.error(
        f"This app can't start: the package(s) **{', '.join(_missing_core)}** aren't installed.\n\n"
        "This means `requirements.txt` wasn't picked up for this deployment. To fix it:\n\n"
        "1. In your GitHub repo, confirm a file literally named `requirements.txt` "
        "(not `requirements (1).txt` or similar) sits in the **same folder** as "
        "`streamlit_app.py`, and that it lists: streamlit, requests, beautifulsoup4, "
        "lxml, pandas, numpy, matplotlib, yfinance.\n"
        "2. In Streamlit Cloud, open this app -> **Manage app** -> **Settings** -> check "
        "**Main file path** points to the exact filename `streamlit_app.py` (no `(1)`/`(2)` suffix).\n"
        "3. Delete any duplicate `streamlit_app (N).py` files in the repo — keep only one.\n"
        "4. Click **Reboot app** (a plain refresh won't reinstall dependencies)."
    )
    st.stop()

# matplotlib is only needed for the chart functions — if the deployment
# environment is missing it (e.g. requirements.txt wasn't picked up),
# the app should still run and screen stocks; charts just get skipped
# with a clear on-screen note instead of the whole app crashing at
# import time (which is what a bare `import matplotlib` at module level
# would do).
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

try:
    import yfinance as yf
except ImportError:
    yf = None


def make_soup(html: str) -> BeautifulSoup:
    """lxml is faster but occasionally fails to install on constrained
    hosts; html.parser ships with Python itself, so fall back to it
    rather than crashing the whole app over a parser choice."""
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


# =============================================================================
# CONFIG
# =============================================================================
PEG_THRESHOLD = 1.5
REJECT_NEGATIVE_PEG = True

STABLE_FLUCTUATION_BAND = 0.125
EXCEPTIONAL_SWING_THRESHOLD = 0.25
COVID_EXCEPTION_YEARS = {2020, 2021}
MAX_NEGATIVE_PAT_YEARS = 3
RECENT_WINDOW_YEARS = 5
LONG_WINDOW_YEARS = 10

# Current/Quick ratio are listed here for display (to match the source
# file), but are only ever populated on a best-effort basis — see
# get_ratios(). They're excluded from ranking if too many companies are
# missing them (handled in rank_companies).
RATIO_DEFINITIONS = {
    "Current Ratio": "higher",
    "Quick Ratio": "higher",
    "Debt/Equity Ratio": "lower",
    "Interest Coverage Ratio": "higher",
    "Return on Equity Ratio": "higher",
    "Net Profit Margin Ratio": "higher",
}
RELIABLY_FREE_RATIOS = {"Debt/Equity Ratio", "Interest Coverage Ratio", "Return on Equity Ratio", "Net Profit Margin Ratio"}

PRICE_HISTORY_YEARS = 10
MIN_PRICE_HISTORY_YEARS = 5
FORECAST_HORIZON_YEARS = 3
NIFTY50_TICKER = "^NSEI"

SECTOR_INDEX_TICKERS = {
    "NIFTY REALTY": "^CNXREALTY", "NIFTY BANK": "^NSEBANK", "NIFTY IT": "^CNXIT",
    "NIFTY AUTO": "^CNXAUTO", "NIFTY PHARMA": "^CNXPHARMA", "NIFTY FMCG": "^CNXFMCG",
    "NIFTY METAL": "^CNXMETAL", "NIFTY ENERGY": "^CNXENERGY",
}

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
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
# SCREENER.IN SCRAPER
# =============================================================================
SCREENER_BASE = "https://www.screener.in"
SCREENER_SEARCH_API = f"{SCREENER_BASE}/api/company/search/"


def _num(text):
    if text is None:
        return None
    m = re.search(r"-?\d+\.?\d*", text.replace(",", "").strip())
    return float(m.group()) if m else None


def _symbol_from_url(url_path):
    """screener.in company URLs are /company/<SYMBOL>/... and <SYMBOL> is
    almost always the real NSE trading symbol — this is the fix for the
    v1 ticker bug (never guess a ticker from the display name when this
    is available)."""
    m = re.search(r"/company/([^/]+)/", url_path)
    return m.group(1).upper() if m else None


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


def screener_fetch_page_by_path(url_path):
    path = url_path.rstrip("/") + "/"
    for u in (f"{SCREENER_BASE}{path}consolidated/", f"{SCREENER_BASE}{path}"):
        try:
            return make_soup(http_get(u)), u
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
    """Returns [{'name':, 'pe':, 'symbol':, 'url':}, ...]. Capturing each
    peer's own screener URL/symbol here (via the row's <a href>) means we
    never have to guess a peer's ticker from its display name later."""
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
        name_cell = tds[name_idx]
        name = name_cell.get_text(strip=True)
        anchor = name_cell.find("a")
        symbol, url = None, None
        if anchor and anchor.get("href"):
            url = anchor["href"]
            symbol = _symbol_from_url(url)
        pe = _num(tds[pe_idx].get_text(strip=True))
        if name:
            peers.append({"name": name, "pe": pe, "symbol": symbol, "url": url})
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
    return {
        "sales": _parse_yearly_row(table, ["Sales", "Revenue", "Sales +"]),
        "net_profit": _parse_yearly_row(table, ["Net Profit", "Net Profit +"]),
        "operating_profit": _parse_yearly_row(table, ["Operating Profit", "Operating Profit +"]),
        "interest": _parse_yearly_row(table, ["Interest"]),
    }


def screener_parse_balance_sheet(soup):
    section = soup.find(id="balance-sheet")
    table = section.find("table") if section else None
    return {
        "equity_capital": _parse_yearly_row(table, ["Equity Capital"]),
        "reserves": _parse_yearly_row(table, ["Reserves"]),
        "borrowings": _parse_yearly_row(table, ["Borrowings"]),
    }


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


def get_screener_snapshot_by_path(url_path, display_name=None):
    soup, used_url = screener_fetch_page_by_path(url_path)
    overview = screener_parse_overview(soup)
    pl = screener_parse_profit_loss(soup)
    bs = screener_parse_balance_sheet(soup)
    return {
        "name": display_name or url_path.strip("/").split("/")[-1],
        "symbol": _symbol_from_url(used_url),
        "url": used_url,
        "overview": overview,
        "stock_pe": screener_get_stock_pe(overview),
        "peers": screener_parse_peers(soup),
        "sales_by_year": pl["sales"],
        "net_profit_by_year": pl["net_profit"],
        "operating_profit_by_year": pl["operating_profit"],
        "interest_by_year": pl["interest"],
        "equity_capital_by_year": bs["equity_capital"],
        "reserves_by_year": bs["reserves"],
        "borrowings_by_year": bs["borrowings"],
        "compounded_growth": screener_parse_compounded_growth(soup),
    }


def get_screener_snapshot(name):
    candidates = screener_search(name)
    match = _best_match(name, candidates)
    if not match:
        raise ValueError(f"No screener.in match found for '{name}'")
    return get_screener_snapshot_by_path(match["url"], display_name=match["name"])


# =============================================================================
# RATIOS — primary source is screener.in's own P&L/balance-sheet data
# (reliable, already scraped). moneycontrol is only a best-effort bonus
# for Current/Quick Ratio, which can't be derived from screener's free
# tier at all (no current-assets/liabilities split published).
# =============================================================================
def _latest_value(values_by_year):
    series = []
    for label, val in values_by_year.items():
        if val is None or label.strip().upper() == "TTM":
            continue
        m = re.search(r"(20\d{2}|19\d{2})", label)
        if m:
            series.append((int(m.group(1)), val))
    if not series:
        return None
    series.sort(key=lambda t: t[0])
    return series[-1][1]


def compute_ratios_from_screener(snap) -> dict:
    ratios = {}
    ov = snap["overview"]
    equity_cap = _latest_value(snap["equity_capital_by_year"])
    reserves = _latest_value(snap["reserves_by_year"])
    borrowings = _latest_value(snap["borrowings_by_year"])
    net_profit = _latest_value(snap["net_profit_by_year"])
    sales = _latest_value(snap["sales_by_year"])
    op_profit = _latest_value(snap["operating_profit_by_year"])
    interest = _latest_value(snap["interest_by_year"])

    equity = (equity_cap + reserves) if (equity_cap is not None and reserves is not None) else None

    if "ROE" in ov:
        ratios["Return on Equity Ratio"] = ov["ROE"]
    elif equity and net_profit is not None:
        ratios["Return on Equity Ratio"] = round(net_profit / equity * 100, 2)

    if borrowings is not None and equity:
        ratios["Debt/Equity Ratio"] = round(borrowings / equity, 2)

    if op_profit is not None and interest:
        ratios["Interest Coverage Ratio"] = round(op_profit / interest, 2)

    if net_profit is not None and sales:
        ratios["Net Profit Margin Ratio"] = round(net_profit / sales * 100, 2)

    return ratios


MC_AUTOSUGGEST_URL = "https://www.moneycontrol.com/mccode/common/autosuggesion.php"
MC_RATIO_LABEL_MAP = {"Current Ratio": ["Current Ratio"], "Quick Ratio": ["Quick Ratio"]}


def _mc_bonus_current_quick_ratio(name):
    """Best-effort only. Returns {} on any failure — callers must not
    depend on this for anything else."""
    raw = http_get(MC_AUTOSUGGEST_URL, params={"query": name, "type": "1", "format": "json"}, as_json=True)
    items = raw if isinstance(raw, list) else raw.get("data", []) if isinstance(raw, dict) else []
    candidates = [{"name": i.get("stock_name") or i.get("name"), "url": i.get("link_src") or i.get("url")} for i in items]
    candidates = [c for c in candidates if c["name"] and c["url"]]
    match = _best_match(name, candidates)
    if not match:
        return {}
    m = re.search(r"stockpricequote/[^/]+/([^/]+)/([A-Za-z0-9]+)$", match["url"])
    if not m:
        return {}
    ratios_url = f"https://www.moneycontrol.com/financials/{m.group(1)}/ratiosVI/{m.group(2)}"
    soup = make_soup(http_get(ratios_url))
    table = soup.find("table", class_=re.compile("mctable|table")) or soup.find("table")
    if not table:
        return {}
    raw_ratios = {}
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        label = cells[0].get_text(strip=True)
        value = None
        for td in cells[1:]:
            mm = re.search(r"-?\d+\.?\d*", td.get_text(strip=True).replace(",", ""))
            if mm:
                value = float(mm.group())
                break
        if label and value is not None:
            raw_ratios[label] = value
    out = {}
    for wanted, variants in MC_RATIO_LABEL_MAP.items():
        for v in variants:
            if v in raw_ratios:
                out[wanted] = raw_ratios[v]
                break
    return out


def get_ratios(name, snapshot=None):
    snap = snapshot or get_screener_snapshot(name)
    ratios = compute_ratios_from_screener(snap)
    source = "screener.in (Debt/Equity, Interest Coverage, ROE, Net Profit Margin)"
    try:
        bonus = _mc_bonus_current_quick_ratio(name)
        if bonus:
            ratios.update(bonus)
            source += " + moneycontrol (Current/Quick Ratio)"
    except Exception:
        pass
    return ratios, source


def rank_companies(ratios_by_company: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(ratios_by_company, orient="index")
    df = df.reindex(columns=list(RATIO_DEFINITIONS.keys()))

    # Only rank on columns where at least half the companies have data —
    # a column that's almost entirely missing (typically Current/Quick
    # Ratio, since free sources rarely publish them) would otherwise
    # produce a meaningless rank.
    usable_cols = [c for c in RATIO_DEFINITIONS if df[c].notna().sum() >= max(1, len(df) // 2)]

    rank_df = pd.DataFrame(index=df.index)
    for ratio in usable_cols:
        direction = RATIO_DEFINITIONS[ratio]
        rank_df[ratio + " Rank"] = df[ratio].rank(ascending=(direction == "lower"), method="average")
    rank_df["Sum Of Ranks"] = rank_df.sum(axis=1, skipna=True) if usable_cols else np.nan
    rank_df["Final Rank"] = rank_df["Sum Of Ranks"].rank(ascending=True, method="min")
    if rank_df["Final Rank"].notna().all():
        rank_df["Final Rank"] = rank_df["Final Rank"].astype(int)

    combined = pd.concat([df, rank_df], axis=1)
    unavailable = [c for c in RATIO_DEFINITIONS if c not in usable_cols]
    return combined.sort_values("Final Rank"), unavailable


# =============================================================================
# STEP 2 & 3 — valuation classification + PEG
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
# STEP 4 — top-line/bottom-line trend analysis
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
# STEPS 6-8 — forecasting. Ticker resolution now uses the real NSE symbol
# captured from screener.in's own URLs (the v1 bug fix) instead of
# guessing from the display name.
# =============================================================================
def get_price_history(nse_symbol: str, years=PRICE_HISTORY_YEARS) -> pd.Series:
    if yf is None:
        raise RuntimeError("yfinance is not installed")
    if not nse_symbol:
        raise ValueError("No ticker symbol available for this company")
    last_err = None
    for suffix in (".NS", ".BO"):
        ticker = f"{nse_symbol}{suffix}"
        try:
            df = yf.download(ticker, period=f"{years}y", interval="1d", progress=False, auto_adjust=True)
            if df.empty or len(df) < MIN_PRICE_HISTORY_YEARS * 150:
                df = yf.download(ticker, period=f"{MIN_PRICE_HISTORY_YEARS}y", interval="1d", progress=False, auto_adjust=True)
            if not df.empty:
                close = df["Close"].dropna()
                close.name = nse_symbol
                return close
        except Exception as e:
            last_err = e
    raise ValueError(f"No price history found for {nse_symbol} (tried .NS and .BO): {last_err}")


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
    ax.fill_between(future_dates, forecast["future_price_lo"], forecast["future_price_hi"], color="#d62728", alpha=0.15, label="Volatility band (not a real forecast interval)")
    ax.set_title(f"{company_label}: price history & trend projection\n(illustrative only — not a prediction)")
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


def plot_revenue_profit_trend(sales_by_year, net_profit_by_year, company_label):
    sales_series = _sorted_series(sales_by_year)
    profit_series = _sorted_series(net_profit_by_year)
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    if sales_series:
        years, vals = zip(*sales_series)
        ax1.bar(years, vals, color="#1f77b4", alpha=0.6, label="Sales")
    ax1.set_ylabel("Sales (Rs. Cr.)", color="#1f77b4")
    ax2 = ax1.twinx()
    if profit_series:
        years2, vals2 = zip(*profit_series)
        ax2.plot(years2, vals2, color="#d62728", marker="o", label="Net Profit")
    ax2.set_ylabel("Net Profit (Rs. Cr.)", color="#d62728")
    ax2.axhline(0, color="#d62728", linewidth=0.5, linestyle=":")
    ax1.set_title(f"{company_label}: revenue & net profit trend")
    fig.tight_layout()
    return fig


# =============================================================================
# ORCHESTRATION
# =============================================================================
def classify_one_company(name, url_path=None):
    """Runs steps 1-4 for one company. Stores the full snapshot in the
    result so downstream ratio/forecast steps can reuse it instead of
    re-scraping the same page again."""
    result = {"name": name, "selected": False, "error": None, "snapshot": None}
    try:
        snap = get_screener_snapshot_by_path(url_path, display_name=name) if url_path else get_screener_snapshot(name)
        result["snapshot"] = snap
        result["symbol"] = snap["symbol"]
        pe = snap["stock_pe"]
        result["pe"] = pe
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
# STREAMLIT UI — simple by default (verdict + charts up front), full
# working tucked into expanders for anyone who wants the detail.
# =============================================================================
st.set_page_config(page_title="Nifty Sector Stock Screener", layout="centered")
st.title("📊 Sector Stock Screener")
st.caption("Enter an NSE-listed company name. Data-driven screen, not investment advice.")

company_name = st.text_input("Company name (as on screener.in)", placeholder="e.g. DLF, Lodha, Godrej Properties")
run_clicked = st.button("Analyze", type="primary")

if run_clicked and company_name.strip():
    try:
        with st.spinner("Looking up the company and its sector peers..."):
            snap = get_screener_snapshot(company_name)

        with st.spinner("Classifying valuation and applying the PEG/trend rule..."):
            main_result = classify_one_company(snap["name"], url_path=snap["url"].replace(SCREENER_BASE, ""))
            main_result["snapshot"] = snap  # already fetched above; avoid a duplicate fetch

        if main_result["error"]:
            st.error(f"Couldn't complete the analysis: {main_result['error']}")
        else:
            # ---- Verdict card, first thing the user sees -----------------
            verdict_ok = main_result["selected"]
            box = st.success if verdict_ok else st.error
            box(
                f"### {'✅ SELECTED' if verdict_ok else '❌ NOT SELECTED'}: {snap['name']}\n"
                f"P/E {main_result['pe']} vs sector median {main_result['sector_median_pe']:.1f} "
                f"→ **{main_result['bucket'].upper()}** → {main_result['reason']}"
            )

            # ---- Charts, shown directly (not hidden) ----------------------
            st.subheader("Charts")
            if not MATPLOTLIB_OK:
                st.info(
                    "Charts are unavailable — matplotlib isn't installed in this deployment "
                    "(check that requirements.txt is committed alongside streamlit_app.py). "
                    "The rest of the analysis below still works."
                )

            if MATPLOTLIB_OK:
                st.pyplot(plot_revenue_profit_trend(snap["sales_by_year"], snap["net_profit_by_year"], snap["name"]))

            price_series = None
            try:
                price_series = get_price_history(snap["symbol"])
                if MATPLOTLIB_OK:
                    st.pyplot(plot_price_with_forecast(price_series, snap["name"]))
            except Exception as e:
                st.warning(f"Couldn't load price history for {snap['name']} ({snap.get('symbol') or 'no ticker found'}): {e}")

            # ---- Step 5: sector ratio ranking (detail, collapsible) ------
            selected_names, ranked_df, unavailable_ratios = [], None, []
            with st.expander("Step 5 — sector ratio ranking", expanded=False):
                with st.spinner("Screening the rest of the sector..."):
                    sector_results = {snap["name"]: main_result}
                    for peer in snap["peers"]:
                        if peer["name"].strip().lower() == snap["name"].strip().lower():
                            continue
                        peer_path = peer["url"] if peer.get("url") else None
                        sector_results[peer["name"]] = classify_one_company(peer["name"], url_path=peer_path)
                        time.sleep(1)
                    selected_names = [n for n, r in sector_results.items() if r.get("selected")]

                st.write("Selected stocks (passed the valuation/PEG/trend screen):", ", ".join(selected_names) or "none")

                ratio_data = {}
                with st.spinner("Computing ratios for selected stocks..."):
                    for c in selected_names:
                        snap_c = sector_results[c].get("snapshot")
                        r, _src = get_ratios(c, snapshot=snap_c)
                        if r:
                            ratio_data[c] = r
                        time.sleep(0.5)

                if ratio_data:
                    ranked_df, unavailable_ratios = rank_companies(ratio_data)
                    st.dataframe(ranked_df)
                    if unavailable_ratios:
                        st.caption(
                            f"⚠️ {', '.join(unavailable_ratios)} aren't reliably published by free sources "
                            "(screener.in doesn't break out current assets/liabilities) and were left out of the ranking."
                        )
                else:
                    st.warning("No ratio data could be retrieved for the selected stocks.")

            # ---- Steps 6-8: forecasting comparison ------------------------
            st.subheader("Forecast vs sector & Nifty 50")
            st.caption("Trend extrapolation of past prices — illustrative only, not a prediction.")
            try:
                with st.spinner("Building the comparison..."):
                    symbol_lookup = {snap["name"]: snap["symbol"]}
                    for p in snap["peers"]:
                        symbol_lookup[p["name"]] = p.get("symbol")

                    peer_symbols = [p["symbol"] for p in snap["peers"] if p.get("symbol") and p["name"].strip().lower() != snap["name"].strip().lower()]
                    sector_series, sector_source = get_sector_benchmark_series(None, peer_symbols)
                    nifty_series = get_nifty50_series()

                    comparison_entries = {}
                    if price_series is not None:
                        comparison_entries[snap["name"]] = price_series
                    comparison_entries[f"Sector ({sector_source})"] = sector_series
                    comparison_entries["Nifty 50"] = nifty_series

                    if ranked_df is not None:
                        for tp in [c for c in ranked_df.index[:2] if c != snap["name"]]:
                            sym = symbol_lookup.get(tp)
                            if sym:
                                try:
                                    comparison_entries[tp] = get_price_history(sym)
                                except Exception:
                                    pass

                    comp_df = build_cagr_comparison(comparison_entries)
                    own_row = comp_df.loc[snap["name"]] if snap["name"] in comp_df.index else None
                    if own_row is not None:
                        beats_own = own_row["Trend-Projected Forward CAGR"] > own_row["Historical CAGR"]
                        st.write("✅ Trend beats its own historical return" if beats_own else "❌ Trend does not beat its own historical return")
                    st.dataframe(comp_df.style.format("{:.1%}"))
                    if MATPLOTLIB_OK:
                        st.pyplot(plot_cagr_bars(comp_df, f"{snap['name']} vs sector, Nifty 50 & top sector performers"))
            except Exception as e:
                st.warning(f"Couldn't build the full benchmark comparison: {e}")

            with st.expander("Technical details"):
                st.json({k: v for k, v in main_result.items() if k != "snapshot"})

        st.caption("This is a data-driven screen, not investment advice — do your own further diligence.")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())

elif run_clicked:
    st.warning("Enter a company name first.")
