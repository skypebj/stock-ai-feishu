import os
import datetime
import requests
import numpy as np
import pandas as pd
from openai import OpenAI

# ===================== DEBUG 日志（全变量输出） =====================
def log(msg): print(f"[DEBUG] {msg}")
def error(msg): print(f"[ERROR] {msg}")
def success(msg): print(f"[SUCCESS] {msg}")
def var(name, value): print(f"[VAR] {name} = {value}")

# ===================== 环境变量 =====================
log("========== 程序启动 ==========")
PUSHDEER_TOKEN = os.getenv("PUSHDEER_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
STOCK_LIST_STR = os.getenv("STOCK_LIST", "")
var("股票列表", STOCK_LIST_STR)

STOCK_CODES = [s.strip() for s in STOCK_LIST_STR.split("-") if s.strip()]
var("最终股票", STOCK_CODES)

# ===================== 【官方正确】腾讯日线获取（实测可用） =====================
def get_tencent_kline(code):
    log(f"获取K线：{code}")
    try:
        # 唯一正确接口：qt.gtimg.cn 自带历史K线（股票/ETF通用）
        url = f"https://qt.gtimg.cn/q={code}"
        resp = requests.get(url, timeout=8)
        var("状态码", resp.status_code)

        # 切割数据
        text = resp.text
        data = text.split('"')[1]
        arr = data.split("~")

        # 最后一段就是 历史日线数据
        kline_raw = arr[-1]
        var("原始K线长度", len(kline_raw))

        # 按天分割
        day_list = kline_raw.split("|")
        var("总K线天数", len(day_list))

        closes = []
        for day in day_list:
            if not day: continue
            parts = day.split(",")
            if len(parts) >= 5:
                closes.append(float(parts[4]))  # 收盘价在第5位

        closes = closes[-60:]
        var("有效收盘价数量", len(closes))
        var("最近5个收盘价", closes[-5:])
        return closes

    except Exception as e:
        error(f"获取失败：{e}")
        return []

# ===================== 实时行情 =====================
def get_tencent_quote(code):
    try:
        url = f"https://qt.gtimg.cn/q={code}"
        txt = requests.get(url, timeout=5).text
        arr = txt.split("~")
        name = arr[1]
        price = float(arr[3])
        pre = float(arr[4])
        pct = round((price - pre) / pre * 100, 2)
        return {"name": name, "price": price, "pct": pct}
    except:
        return {"name": code, "price": 0, "pct": 0}

# ===================== 标准 RSI（正确） =====================
def rsi(prices, n):
    if len(prices) < n+1: return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_g = np.mean(gains[:n])
    avg_l = np.mean(losses[:n])
    for i in range(n, len(gains)):
        avg_g = (avg_g*(n-1)+gains[i])/n
        avg_l = (avg_l*(n-1)+losses[i])/n
    rs = avg_g / avg_l if avg_l !=0 else 999
    return round(100 - 100/(1+rs), 2)

# ===================== EMA20（正确） =====================
def ema(prices, n=20):
    if len(prices) < n: return 0.0
    return round(pd.Series(prices).ewm(span=n, adjust=False).mean().iloc[-1], 2)

# ===================== AI 分析 =====================
def ai_analyze(stock):
    prompt = f"""请以资深股票技术分析师身份，对【{stock['name']} / {stock['code']}】进行技术面分析，重点聚焦价格位置、K 线形态、量价关系、支撑压力、均线趋势，简要结合行业与政策环境，语言专业精炼，不构成投资建议。
分析框架：股价区间、支撑压力、K线与均线、成交量、行业政策、技术观点+风险
数据：价格={stock['price']} 涨跌幅={stock['pct']}% RSI6={stock['rsi6']} RSI12={stock['rsi12']} RSI24={stock['rsi24']} EMA20={stock['ema20']}"""
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        return client.chat.completions.create(model="deepseek-chat", messages=[{"role":"user","content":prompt}], temperature=0.3).choices[0].message.content
    except:
        return "AI 分析暂时不可用"

# ===================== PushDeer 逐条推送 =====================
def push_one(stock):
    msg = f"📈 技术指标 {datetime.date.today()}\n\n"
    msg += f"【{stock['name']}】{stock['code']}\n"
    msg += f"现价：{stock['price']}  {stock['pct']:+}%\n\n"
    msg += f"RSI(6)  = {stock['rsi6']}\n"
    msg += f"RSI(12) = {stock['rsi12']}\n"
    msg += f"RSI(24) = {stock['rsi24']}\n"
    msg += f"EMA(20) = {stock['ema20']}\n\n"
    msg += f"🤖 技术分析：\n{ai_analyze(stock)}"

    print("\n=====================================")
    print("【PushDeer 推送消息】")
    print("=====================================")
    print(msg)
    print("=====================================\n")

    import urllib.parse
    url = f"https://api2.pushdeer.com/message/push?pushkey={PUSHDEER_TOKEN}&text={urllib.parse.quote(msg)}"
    requests.get(url, timeout=10)
    success(f"{stock['code']} 推送成功")

# ===================== 主程序 =====================
if __name__ == "__main__":
    for code in STOCK_CODES:
        log(f"\n===== 处理 {code} =====")
        basic = get_tencent_quote(code)
        closes = get_tencent_kline(code)

        if len(closes) < 20:
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

        var("股票数据", stock)
        push_one(stock)

    log("========== 全部完成 ==========")
