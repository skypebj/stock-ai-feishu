import os
import sys
import datetime
import numpy as np
import pandas as pd
import akshare as ak
from openai import OpenAI

# ===================== 超级 DEBUG 日志（变量全打印） =====================
def log(msg):
    print(f"[DEBUG] {msg}")

def error(msg):
    print(f"[ERROR] {msg}")

def success(msg):
    print(f"[SUCCESS] {msg}")

def var(name, value):
    print(f"[VAR] {name} = {value}")

# ===================== 环境变量 =====================
log("========== 程序启动 ==========")
PUSHDEER_TOKEN = os.getenv("PUSHDEER_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
STOCK_LIST_STR = os.getenv("STOCK_LIST", "")

var("股票原始字符串", STOCK_LIST_STR)
STOCK_CODES = [s.strip() for s in STOCK_LIST_STR.split("-") if s.strip()]
var("最终股票列表", STOCK_CODES)

# ===================== AKShare 获取日线（标准、稳定） =====================
def get_ak_daily(code):
    log(f"--- AKShare 获取日线：{code} ---")
    try:
        # 转换代码：sh601328 → 601328.SH
        symbol = code[2:] + "." + code[:2].upper()
        var("akshare 标准代码", symbol)

        # 获取日线（前复权）
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
        var("获取K线行数", len(df))
        var("最近K线", df[["收盘"]].tail(10).to_string())

        closes = df["收盘"].tolist()
        return closes[-60:]  # 取最近60日

    except Exception as e:
        error(f"AKShare 获取失败：{e}")
        return []

# ===================== 实时名称 & 价格 =====================
def get_ak_basic(code):
    try:
        symbol = code[2:] + "." + code[:2].upper()
        df = ak.stock_zh_a_spot_em(symbol=symbol)
        name = df.iloc[0]["名称"]
        price = df.iloc[0]["最新"]
        pct = df.iloc[0]["涨跌幅"]
        return {"name": name, "price": round(price,2), "pct": round(pct,2)}
    except:
        return {"name": code, "price":0, "pct":0}

# ===================== 标准 RSI（Wilder 算法） =====================
def rsi(prices, n):
    log(f"--- 计算 RSI({n}) ---")
    var("K线数量", len(prices))
    if len(prices) < n+1: return 50.0

    deltas = np.diff(prices)
    gains = np.where(deltas>0, deltas, 0)
    losses = np.where(deltas<0, -deltas, 0)

    avg_g = np.mean(gains[:n])
    avg_l = np.mean(losses[:n])
    for i in range(n, len(gains)):
        avg_g = (avg_g*(n-1)+gains[i])/n
        avg_l = (avg_l*(n-1)+losses[i])/n

    rs = avg_g/avg_l if avg_l !=0 else 999
    res = 100 - 100/(1+rs)
    var(f"RSI({n})", round(res,2))
    return round(res,2)

# ===================== EMA20 =====================
def ema(prices, n=20):
    if len(prices)<n: return 0.0
    return round(pd.Series(prices).ewm(span=n, adjust=False).mean().iloc[-1], 2)

# ===================== AI 分析（你指定的完整提示词） =====================
def ai_analyze(stock):
    prompt = f"""请以资深股票技术分析师身份，对【{stock['name']} / {stock['code']}】进行技术面分析，重点聚焦价格位置、K 线形态、量价关系、支撑压力、均线趋势，简要结合行业与政策环境，语言专业精炼，不构成投资建议。
分析框架：
股价当前所处区间（高位/中位/低位），关键支撑位与压力位
K 线形态、趋势结构、均线系统表现
成交量配合情况
行业景气度及相关政策简要影响
技术面观点 + 风险提示

数据：价格={stock['price']} 涨跌幅={stock['pct']}% RSI6={stock['rsi6']} RSI12={stock['rsi12']} RSI24={stock['rsi24']} EMA20={stock['ema20']}
"""
    try:
        c = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        return c.chat.completions.create(model="deepseek-chat", messages=[{"role":"user","content":prompt}], temperature=0.3).choices[0].message.content
    except:
        return "AI 分析暂时不可用"

# ===================== PushDeer 逐股推送（每条单独发） =====================
def push_one(stock):
    msg = f"📈 技术指标 {datetime.date.today()}\n\n"
    msg += f"【{stock['name']}】{stock['code']}\n"
    msg += f"现价：{stock['price']}  {stock['pct']:+}%\n\n"
    msg += f"RSI(6)  = {stock['rsi6']}\n"
    msg += f"RSI(12) = {stock['rsi12']}\n"
    msg += f"RSI(24) = {stock['rsi24']}\n"
    msg += f"EMA(20) = {stock['ema20']}\n\n"
    msg += f"🤖 技术分析：\n{ai_analyze(stock)}"

    # 【回显推送内容】
    print("\n=====================================")
    print("【PushDeer 推送消息】")
    print("=====================================")
    print(msg)
    print("=====================================\n")

    import urllib.parse
    url = f"https://api2.pushdeer.com/message/push?pushkey={PUSHDEER_TOKEN}&text={urllib.parse.quote(msg)}"
    import requests
    requests.get(url, timeout=10)
    success(f"{stock['code']} 推送完成")

# ===================== 主程序 =====================
if __name__ == "__main__":
    for code in STOCK_CODES:
        log(f"\n===== 处理 {code} =====")
        basic = get_ak_basic(code)
        closes = get_ak_daily(code)

        if len(closes) < 30:
            error(f"{code} 数据不足")
            continue

        stock = {
            "code": code,
            "name": basic["name"],
            "price": basic["price"],
            "pct": basic["pct"],
            "rsi6": rsi(closes,6),
            "rsi12": rsi(closes,12),
            "rsi24": rsi(closes,24),
            "ema20": ema(closes,20)
        }

        var("最终股票数据", stock)
        push_one(stock)

    log("========== 全部完成 ==========")
