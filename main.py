import os
import datetime
import requests
import numpy as np
import pandas as pd
from openai import OpenAI

# ===================== 超级 DEBUG =====================
def log(msg): print(f"[DEBUG] {msg}")
def error(msg): print(f"[ERROR] {msg}")
def var(name, value): print(f"[VAR] {name} = {value}")

# ===================== 环境变量 =====================
log("========== 程序启动 ==========")
PUSHDEER_TOKEN = os.getenv("PUSHDEER_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
STOCK_LIST_STR = os.getenv("STOCK_LIST", "")
var("股票列表", STOCK_LIST_STR)

STOCK_CODES = [s.strip() for s in STOCK_LIST_STR.split("-") if s.strip()]
var("最终股票", STOCK_CODES)

# ===================== 【最强调试】获取K线（打印全部内容） =====================
def get_tencent_kline(code):
    log(f"获取K线：{code}")
    try:
        url = f"https://qt.gtimg.cn/q={code}"
        var("请求URL", url)

        resp = requests.get(url, timeout=8)
        var("状态码", resp.status_code)
        var("响应长度", len(resp.text))
        var("完整响应内容", resp.text.strip())  # 直接打印接口返回的全部内容！

        # 尝试解析
        if '"' not in resp.text:
            error("返回内容不包含双引号，无数据！")
            return []

        data = resp.text.split('"')[1]
        var("双引号内数据", data)
        arr = data.split("~")
        var("分割后字段数量", len(arr))

        # 打印所有字段
        for i, part in enumerate(arr):
            var(f"字段[{i}]", part[:50] + "..." if len(part) > 50 else part)

        # 取最后一段
        kline_raw = arr[-1] if len(arr) > 0 else ""
        var("最后一段K线数据", kline_raw[:100])

        day_list = kline_raw.split("|")
        var("按|分割后的天数", len(day_list))

        closes = []
        for day in day_list:
            if not day: continue
            parts = day.split(",")
            if len(parts) >= 5:
                closes.append(float(parts[4]))

        closes = closes[-60:]
        var("最终有效收盘价数量", len(closes))
        return closes

    except Exception as e:
        error(f"异常：{e}")
        return []

# ===================== 实时行情 =====================
def get_tencent_quote(code):
    try:
        url = f"https://qt.gtimg.cn/q={code}"
        txt = requests.get(url, timeout=5).text
        arr = txt.split("~")
        name = arr[1] if len(arr) > 1 else code
        price = float(arr[3]) if len(arr) > 3 else 0.0
        pre = float(arr[4]) if len(arr) > 4 else 0.0
        pct = round((price - pre) / pre * 100, 2) if pre != 0 else 0
        return {"name": name, "price": price, "pct": pct}
    except:
        return {"name": code, "price": 0, "pct": 0}

# ===================== RSI & EMA =====================
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

def ema(prices, n=20):
    if len(prices) < n: return 0.0
    return round(pd.Series(prices).ewm(span=n, adjust=False).mean().iloc[-1], 2)

# ===================== AI & PushDeer =====================
def ai_analyze(stock):
    return "AI 调试中"

def push_one(stock):
    log(f"推送：{stock}")

# ===================== 主程序 =====================
if __name__ == "__main__":
    for code in STOCK_CODES:
        log(f"\n===== 处理 {code} =====")
        get_tencent_kline(code)

    log("========== 调试结束 ==========")
