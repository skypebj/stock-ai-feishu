import os
import sys
import requests
import datetime
import numpy as np
import pandas as pd
from openai import OpenAI

# ===================== 超级日志 =====================
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

var("PUSHDEER_TOKEN 是否存在", PUSHDEER_TOKEN is not None)
var("DEEPSEEK_API_KEY 是否存在", DEEPSEEK_API_KEY is not None)
var("原始股票字符串", STOCK_LIST_STR)

STOCK_CODES = [s.strip() for s in STOCK_LIST_STR.split("-") if s.strip()]
var("分割后股票列表", STOCK_CODES)

# ===================== 1. 获取日线收盘价 =====================
def get_daily_closes(code):
    log(f"--- 开始获取 {code} 日线数据 ---")
    try:
        url = f"https://data.gtimg.cn/flashdata/hushen/daily/{code[:2]}/{code}.js"
        var("请求URL", url)

        resp = requests.get(url, timeout=12)
        var("HTTP状态码", resp.status_code)
        var("返回内容长度", len(resp.text))

        data_part = resp.text.split('"')[1]
        lines = data_part.split("\\n")
        var("K线总行数", len(lines))

        closes = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    close_val = float(parts[3])
                    closes.append(close_val)
                    if i < 10:
                        var(f"第{i}条收盘价", close_val)
                except:
                    continue

        var("最终有效收盘价数量", len(closes))
        var("最近10个收盘价", closes[-10:])
        return closes[-60:]

    except Exception as e:
        error(f"获取K线失败: {e}")
        return []

# ===================== 2. 标准 RSI 计算（带步骤打印） =====================
def rsi(prices, period):
    log(f"--- 计算 RSI({period}) ---")
    var("输入K线数量", len(prices))
    var("周期", period)

    if len(prices) < period + 1:
        log("数据不足，返回默认 50")
        return 50.0

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    var("deltas（价格变化）", deltas[:10])
    var("gains（上涨）", gains[:10])
    var("losses（下跌）", losses[:10])

    avg_g = np.mean(gains[:period])
    avg_l = np.mean(losses[:period])
    var("初始avg_gain", avg_g)
    var("初始avg_loss", avg_l)

    for i in range(period, len(gains)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period

    var("最终avg_gain", avg_g)
    var("最终avg_loss", avg_l)

    if avg_l == 0:
        var("RSI结果", 100.0)
        return 100.0

    rs = avg_g / avg_l
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    var("RS", rs)
    var("RSI结果", round(rsi_val, 2))

    return round(rsi_val, 2)

# ===================== 3. EMA20 计算（带步骤打印） =====================
def ema(prices, n=20):
    log(f"--- 计算 EMA({n}) ---")
    var("输入K线数量", len(prices))
    var("周期", n)

    if len(prices) < n:
        log("数据不足，返回 0")
        return 0.0

    ema_series = pd.Series(prices).ewm(span=n, adjust=False).mean()
    ema_val = ema_series.iloc[-1]

    var("最近3个EMA值", list(ema_series[-3:]))
    var("最终EMA结果", round(ema_val, 2))

    return round(ema_val, 2)

# ===================== 4. 实时价格名称 =====================
def get_basic(code):
    log(f"--- 获取实时价格 {code} ---")
    try:
        url = f"https://qt.gtimg.cn/q={code}"
        arr = requests.get(url, timeout=6).text.split("~")
        name = arr[1]
        price = float(arr[3])
        pre = float(arr[4])
        pct = round((price - pre) / pre * 100, 2)

        var("股票名称", name)
        var("当前价", price)
        var("昨收价", pre)
        var("涨跌幅", pct)
        return {"name": name, "price": price, "pct": pct}
    except Exception as e:
        error(f"获取实时价格失败: {e}")
        return {"name": code, "price": 0.0, "pct": 0.0}

# ===================== 5. AI 分析 =====================
def ai_analyze(stock):
    log(f"--- AI 分析 {stock['code']} ---")
    prompt = f"""请以资深股票技术分析师身份，对【{stock['name']} / {stock['code']}】进行技术面分析，重点聚焦价格位置、K 线形态、量价关系、支撑压力、均线趋势，简要结合行业与政策环境，语言专业精炼，不构成投资建议。
分析框架：股价区间、支撑压力、K线与均线、成交量、行业政策、技术观点+风险
数据：价格{stock['price']}元，涨跌幅{stock['pct']}%，RSI6={stock['rsi6']}，RSI12={stock['rsi12']}，RSI24={stock['rsi24']}，EMA20={stock['ema20']}"""

    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        res = client.chat.completions.create(model="deepseek-chat", messages=[{"role":"user","content":prompt}], temperature=0.3)
        return res.choices[0].message.content
    except Exception as e:
        return f"AI失败: {e}"

# ===================== 6. PushDeer 单股推送 + 完整回显 =====================
def push_one(stock):
    log(f"--- 推送 {stock['code']} ---")
    msg = f"📈 个股技术指标 {datetime.date.today()}\n\n"
    msg += f"【{stock['name']}】{stock['code']}\n"
    msg += f"现价：{stock['price']} 元  {stock['pct']:+}%\n\n"
    msg += f"RSI(6)  = {stock['rsi6']}\n"
    msg += f"RSI(12) = {stock['rsi12']}\n"
    msg += f"RSI(24) = {stock['rsi24']}\n"
    msg += f"EMA(20) = {stock['ema20']}\n\n"
    msg += f"🤖 技术分析：\n{ai_analyze(stock)}"

    print("\n========================================")
    print("【📌 即将推送到 PushDeer 的完整消息】")
    print("========================================")
    print(msg)
    print("========================================\n")

    try:
        import urllib.parse
        push_url = f"https://api2.pushdeer.com/message/push?pushkey={PUSHDEER_TOKEN}&text={urllib.parse.quote(msg)}"
        requests.get(push_url, timeout=10)
        success(f"{stock['code']} 推送成功")
    except Exception as e:
        error(f"推送失败: {e}")

# ===================== 主程序 =====================
if __name__ == "__main__":
    log("========== 开始执行 ==========")
    for code in STOCK_CODES:
        log(f"\n\n##############################")
        log(f"处理股票：{code}")
        log(f"##############################")

        basic = get_basic(code)
        closes = get_daily_closes(code)

        if len(closes) < 30:
            error(f"{code} 数据不足，跳过")
            continue

        stock = {
            "code": code,
            "name": basic["name"],
            "price": round(basic["price"], 2),
            "pct": basic["pct"],
            "rsi6": rsi(closes, 6),
            "rsi12": rsi(closes, 12),
            "rsi24": rsi(closes, 24),
            "ema20": ema(closes, 20)
        }

        var("最终股票对象", stock)
        push_one(stock)

    log("========== 全部执行完成 ==========")
