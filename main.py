import os
import datetime
import requests
import numpy as np
import pandas as pd
import baostock as bs

# ===================== 核心指标计算 =====================
def rsi(prices, n):
    if len(prices) < n+1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_g = np.mean(gains[:n])
    avg_l = np.mean(losses[:n])
    for i in range(n, len(gains)):
        avg_g = (avg_g * (n-1) + gains[i]) / n
        avg_l = (avg_l * (n-1) + losses[i]) / n
    rs = avg_g / avg_l if avg_l != 0 else 999
    return round(100 - 100/(1+rs), 2)

def ema(prices, n=20):
    if len(prices) < n:
        return 0.0
    return round(pd.Series(prices).ewm(span=n, adjust=False).mean().iloc[-1], 2)

# ===================== 数据获取 =====================
def get_daily(code):
    bs_code = f"{code[:2]}.{code[2:]}"
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
    rs = bs.query_history_k_data_plus(bs_code, "date,close", start_date, end_date, "d", "3")
    df = rs.get_data()
    df["close"] = pd.to_numeric(df["close"])
    return df["close"].dropna().tolist()[-60:]

def get_quote(code):
    try:
        txt = requests.get(f"https://qt.gtimg.cn/q={code}", timeout=5).text
        arr = txt.split("~")
        name = arr[1]
        price = float(arr[3])
        pre = float(arr[4])
        pct = round((price - pre)/pre*100, 2)
        return {"name": name, "price": price, "pct": pct}
    except:
        return {"name": code, "price": 0, "pct": 0}

# ===================== 推送消息（输出到 GitHub 日志 → 触发邮件） =====================
def send_github_notice(stock_list):
    print("\n📊 股票指标分析结果（GitHub 邮件通知）")
    print("="*50)
    for s in stock_list:
        print(f"📈 {s['name']} ({s['code']})")
        print(f"   现价：{s['price']}  {s['pct']:+}%")
        print(f"   RSI6={s['rsi6']}  RSI12={s['rsi12']}  RSI24={s['rsi24']}")
        print(f"   EMA20={s['ema20']}")
        print("-"*30)

# ===================== 主程序 =====================
if __name__ == "__main__":
    bs.login()
    stock_list = os.getenv("STOCK_LIST", "sh513100-sh601328").split("-")
    result = []

    for code in stock_list:
        code = code.strip()
        basic = get_quote(code)
        closes = get_daily(code)
        if len(closes) < 20:
            continue

        item = {
            "code": code,
            "name": basic["name"],
            "price": basic["price"],
            "pct": basic["pct"],
            "rsi6": rsi(closes, 6),
            "rsi12": rsi(closes, 12),
            "rsi24": rsi(closes, 24),
            "ema20": ema(closes, 20)
        }
        result.append(item)

    # 输出结果 → 自动进入 GitHub 邮件
    send_github_notice(result)
    bs.logout()
