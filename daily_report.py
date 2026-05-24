"""
Daily Financial Dashboard - Simon Yun
매일 오전 5시 KST 자동 실행 (GitHub Actions)
"""

import yfinance as yf
import requests
import json
import os
from datetime import datetime, timedelta
import pytz
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np

# ── 설정 ──────────────────────────────────────────────
KST = pytz.timezone('Asia/Seoul')
NOW = datetime.now(KST)
TODAY_STR = NOW.strftime('%Y년 %m월 %d일 (%a)')

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# 지수
INDICES = {
    "S&P 500":   "^GSPC",
    "NASDAQ":    "^IXIC",
    "DOW":       "^DJI",
    "KOSPI":     "^KS11",
    "KOSDAQ":    "^KQ11",
}

# 보유 종목
HOLDINGS = {
    "Tesla":            {"ticker": "TSLA",      "type": "us"},
    "NVIDIA":           {"ticker": "NVDA",      "type": "us"},
    "Koact K수출핵심30": {"ticker": "449450.KS", "type": "kr"},
    "Tiger 미국우주테크": {"ticker": "418660.KS", "type": "kr"},
}

# 코인
COINS = {
    "Bitcoin":  "BTC-USD",
    "Ethereum": "ETH-USD",
    "XRP":      "XRP-USD",
    "Solana":   "SOL-USD",
}

# 환율
FX = {
    "USD/KRW": "KRW=X",
    "USD/CNY": "CNY=X",
    "USD/JPY": "JPY=X",
    "EUR/USD": "EURUSD=X",
}

CHART_COLOR = {
    "up":   "#00C896",
    "down": "#FF4D6D",
    "line": "#4A9EFF",
    "bg":   "#0D1117",
    "grid": "#1C2333",
    "text": "#C9D1D9",
}

# ── 헬퍼 ──────────────────────────────────────────────
def fmt_num(v, digits=2):
    if v is None:
        return "N/A"
    return f"{v:,.{digits}f}"

def fmt_pct(v):
    if v is None:
        return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"

def color_cls(v):
    if v is None:
        return "neutral"
    return "up" if v >= 0 else "down"

def arrow(v):
    if v is None:
        return "–"
    return "▲" if v >= 0 else "▼"

# ── 데이터 수집 ────────────────────────────────────────
def fetch_quote(ticker):
    """단일 티커의 최신 가격 & 전일비 반환"""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        if len(hist) < 2:
            return None, None, None
        close_prev = hist["Close"].iloc[-2]
        close_last = hist["Close"].iloc[-1]
        chg = close_last - close_prev
        pct = chg / close_prev * 100
        return close_last, chg, pct
    except Exception as e:
        print(f"[WARN] fetch_quote({ticker}): {e}")
        return None, None, None

def fetch_2y(ticker):
    """2년치 종가 시리즈 반환"""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="2y")
        return hist["Close"] if not hist.empty else None
    except Exception as e:
        print(f"[WARN] fetch_2y({ticker}): {e}")
        return None

def make_chart_b64(series, title, color=None):
    """matplotlib 차트를 base64 PNG로 반환"""
    if series is None or len(series) < 5:
        return None

    if color is None:
        first, last = series.iloc[0], series.iloc[-1]
        color = CHART_COLOR["up"] if last >= first else CHART_COLOR["down"]

    fig, ax = plt.subplots(figsize=(6, 2.2))
    fig.patch.set_facecolor(CHART_COLOR["bg"])
    ax.set_facecolor(CHART_COLOR["bg"])

    ax.plot(series.index, series.values, color=color, linewidth=1.5, zorder=3)
    ax.fill_between(series.index, series.values, series.values.min(),
                    color=color, alpha=0.12, zorder=2)

    ax.set_title(title, color=CHART_COLOR["text"], fontsize=9, pad=4)
    ax.tick_params(colors=CHART_COLOR["text"], labelsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y.%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}" if x >= 1000 else f"{x:.2f}"))
    for spine in ax.spines.values():
        spine.set_edgecolor(CHART_COLOR["grid"])
    ax.grid(True, color=CHART_COLOR["grid"], linewidth=0.5, zorder=1)
    fig.tight_layout(pad=0.5)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=CHART_COLOR["bg"])
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ── Claude 시황 코멘트 ────────────────────────────────
def get_ai_comment(summary_text):
    if not ANTHROPIC_API_KEY:
        return "⚠️ ANTHROPIC_API_KEY 미설정 — AI 코멘트를 생성할 수 없습니다."
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 600,
                "messages": [{
                    "role": "user",
                    "content": (
                        f"다음은 오늘({NOW.strftime('%Y-%m-%d')}) 금융 시장 데이터입니다.\n\n"
                        f"{summary_text}\n\n"
                        "LG전자 북미 가전사업 담당 임원의 관점에서 "
                        "①미국·한국 시장 전반 흐름, ②보유 종목 주요 이슈, ③환율이 사업에 미치는 영향을 "
                        "간결하게 3~5문장으로 한국어로 요약해줘. "
                        "말투는 간결·전문적으로."
                    )
                }]
            },
            timeout=30
        )
        data = resp.json()
        return data["content"][0]["text"]
    except Exception as e:
        return f"⚠️ AI 코멘트 오류: {e}"

# ── HTML 생성 ──────────────────────────────────────────
def build_html(indices_data, holdings_data, coins_data, fx_data, ai_comment, charts):
    def index_rows():
        rows = ""
        for name, (price, chg, pct) in indices_data.items():
            cls = color_cls(pct)
            rows += f"""
            <tr>
              <td class="name">{name}</td>
              <td class="price">{fmt_num(price)}</td>
              <td class="{cls}">{arrow(pct)} {fmt_num(abs(chg) if chg else None)} ({fmt_pct(pct)})</td>
            </tr>"""
        return rows

    def holding_cards():
        cards = ""
        for name, d in holdings_data.items():
            price, chg, pct = d["price"], d["chg"], d["pct"]
            cls = color_cls(pct)
            img_tag = ""
            if name in charts:
                img_tag = f'<img src="data:image/png;base64,{charts[name]}" class="chart-img" />'
            suffix = "USD" if HOLDINGS[name]["type"] == "us" else "KRW"
            cards += f"""
            <div class="card">
              <div class="card-header">
                <span class="card-name">{name}</span>
                <span class="badge {'badge-us' if HOLDINGS[name]['type']=='us' else 'badge-kr'}">
                  {'🇺🇸 US' if HOLDINGS[name]['type']=='us' else '🇰🇷 KR'}
                </span>
              </div>
              <div class="card-price">{fmt_num(price)} <span class="currency">{suffix}</span></div>
              <div class="card-chg {cls}">{arrow(pct)} {fmt_num(abs(chg) if chg else None)} ({fmt_pct(pct)})</div>
              {img_tag}
            </div>"""
        return cards

    def coin_rows():
        rows = ""
        for name, (price, chg, pct) in coins_data.items():
            cls = color_cls(pct)
            highlight = ' class="highlight-row"' if name == "XRP" else ""
            rows += f"""
            <tr{highlight}>
              <td class="name">{'⭐ ' if name=='XRP' else ''}{name}</td>
              <td class="price">${fmt_num(price)}</td>
              <td class="{cls}">{arrow(pct)} {fmt_pct(pct)}</td>
            </tr>"""
        return rows

    def fx_rows():
        rows = ""
        for pair, (price, chg, pct) in fx_data.items():
            cls = color_cls(pct)
            rows += f"""
            <tr>
              <td class="name">{pair}</td>
              <td class="price">{fmt_num(price, 4)}</td>
              <td class="{cls}">{arrow(pct)} {fmt_pct(pct)}</td>
            </tr>"""
        return rows

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Daily Market Report – {NOW.strftime('%Y.%m.%d')}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@300;400;600&family=JetBrains+Mono:wght@400;600&display=swap');

  :root {{
    --bg:       #0D1117;
    --surface:  #161B22;
    --border:   #21262D;
    --up:       #00C896;
    --down:     #FF4D6D;
    --accent:   #4A9EFF;
    --text:     #E6EDF3;
    --muted:    #7D8590;
    --gold:     #E3B341;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'IBM Plex Sans KR', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 24px 16px 48px;
  }}

  .header {{
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 28px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .header h1 {{
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    color: var(--accent);
  }}
  .header .date {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--muted);
  }}

  .section-title {{
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 28px 0 12px;
    padding-left: 4px;
    border-left: 2px solid var(--accent);
    padding-left: 8px;
  }}

  /* AI 코멘트 */
  .ai-box {{
    background: linear-gradient(135deg, #161B22 60%, #0d2137);
    border: 1px solid #1d3552;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 8px;
    font-size: 0.88rem;
    line-height: 1.7;
    color: #cdd9e5;
    position: relative;
  }}
  .ai-box::before {{
    content: '🤖 AI 시황';
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: var(--accent);
    display: block;
    margin-bottom: 8px;
  }}

  /* 테이블 */
  table {{
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    background: var(--surface);
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
  }}
  th {{
    background: #1C2333;
    color: var(--muted);
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 8px 14px;
    text-align: left;
    font-weight: 600;
    border-bottom: 1px solid var(--border);
  }}
  td {{ padding: 10px 14px; border-bottom: 1px solid var(--border); }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(255,255,255,0.02); }}
  td.name  {{ color: var(--text); font-weight: 500; }}
  td.price {{ color: var(--text); text-align: right; }}
  td.up    {{ color: var(--up);   text-align: right; }}
  td.down  {{ color: var(--down); text-align: right; }}
  td.neutral {{ color: var(--muted); text-align: right; }}
  tr.highlight-row td {{ background: rgba(227,179,65,0.06); }}
  tr.highlight-row td.name {{ color: var(--gold); }}

  /* 보유종목 카드 */
  .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    transition: border-color 0.2s;
  }}
  .card:hover {{ border-color: var(--accent); }}
  .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
  .card-name {{ font-size: 0.82rem; font-weight: 600; }}
  .badge {{ font-size: 0.65rem; padding: 2px 7px; border-radius: 20px; font-weight: 600; }}
  .badge-us {{ background: rgba(74,158,255,0.15); color: var(--accent); }}
  .badge-kr {{ background: rgba(0,200,150,0.12); color: var(--up); }}
  .card-price {{ font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 600; margin-bottom: 4px; }}
  .currency {{ font-size: 0.65rem; color: var(--muted); }}
  .card-chg {{ font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; margin-bottom: 12px; }}
  .card-chg.up   {{ color: var(--up); }}
  .card-chg.down {{ color: var(--down); }}
  .chart-img {{ width: 100%; border-radius: 6px; margin-top: 4px; }}

  .footer {{
    margin-top: 36px;
    text-align: center;
    font-size: 0.7rem;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>📊 Daily Market Report</h1>
  <span class="date">Simon H.K. Yun &nbsp;|&nbsp; {TODAY_STR} &nbsp;|&nbsp; {NOW.strftime('%H:%M')} KST</span>
</div>

<div class="ai-box">{ai_comment}</div>

<div class="section-title">글로벌 주요 지수</div>
<table>
  <thead><tr><th>지수</th><th style="text-align:right">종가</th><th style="text-align:right">전일비</th></tr></thead>
  <tbody>{index_rows()}</tbody>
</table>

<div class="section-title">보유 종목 (2Y 트렌드)</div>
<div class="cards">{holding_cards()}</div>

<div class="section-title">주요 암호화폐</div>
<table>
  <thead><tr><th>코인</th><th style="text-align:right">현재가 (USD)</th><th style="text-align:right">24H 등락</th></tr></thead>
  <tbody>{coin_rows()}</tbody>
</table>

<div class="section-title">주요 환율</div>
<table>
  <thead><tr><th>통화쌍</th><th style="text-align:right">환율</th><th style="text-align:right">전일비</th></tr></thead>
  <tbody>{fx_rows()}</tbody>
</table>

<div class="footer">
  Generated by Claude Agent &nbsp;·&nbsp; Data: Yahoo Finance &nbsp;·&nbsp; {NOW.strftime('%Y-%m-%d %H:%M:%S')} KST
</div>

</body>
</html>"""

# ── 메인 ──────────────────────────────────────────────
def main():
    print(f"[{NOW.strftime('%H:%M:%S')}] Starting daily report generation...")

    # 1) 지수
    indices_data = {}
    for name, ticker in INDICES.items():
        p, c, pct = fetch_quote(ticker)
        indices_data[name] = (p, c, pct)
        print(f"  {name}: {fmt_num(p)} ({fmt_pct(pct)})")

    # 2) 보유 종목
    holdings_data = {}
    charts = {}
    summary_parts = ["[보유 종목]\n"]
    for name, info in HOLDINGS.items():
        p, c, pct = fetch_quote(info["ticker"])
        holdings_data[name] = {"price": p, "chg": c, "pct": pct}
        summary_parts.append(f"  {name}: {fmt_num(p)} ({fmt_pct(pct)})")
        series = fetch_2y(info["ticker"])
        b64 = make_chart_b64(series, f"{name} 2Y")
        if b64:
            charts[name] = b64
        print(f"  {name}: {fmt_num(p)} ({fmt_pct(pct)})")

    # 3) 코인
    coins_data = {}
    summary_parts.append("\n[코인]\n")
    for name, ticker in COINS.items():
        p, c, pct = fetch_quote(ticker)
        coins_data[name] = (p, c, pct)
        summary_parts.append(f"  {name}: ${fmt_num(p)} ({fmt_pct(pct)})")
        print(f"  {name}: ${fmt_num(p)} ({fmt_pct(pct)})")

    # 4) 환율
    fx_data = {}
    summary_parts.append("\n[환율]\n")
    for pair, ticker in FX.items():
        p, c, pct = fetch_quote(ticker)
        fx_data[pair] = (p, c, pct)
        summary_parts.append(f"  {pair}: {fmt_num(p,4)} ({fmt_pct(pct)})")
        print(f"  {pair}: {fmt_num(p,4)} ({fmt_pct(pct)})")

    # 5) AI 코멘트
    summary_text = "\n".join(summary_parts)
    print("  Requesting AI comment...")
    ai_comment = get_ai_comment(summary_text)

    # 6) HTML 저장
    html = build_html(indices_data, holdings_data, coins_data, fx_data, ai_comment, charts)
    out_path = "docs/index.html"
    os.makedirs("docs", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[✓] Report saved → {out_path}")

if __name__ == "__main__":
    main()
