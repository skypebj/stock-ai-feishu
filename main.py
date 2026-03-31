import os
import sys
import requests
import datetime
import numpy as np
from openai import OpenAI

# ===================== 日志 =====================
def log(msg):
    print(f"[DEBUG] {msg}")

def error(msg):
    print(f"[ERROR] {msg}")

def success(msg):
    print(f"[SUCCESS] {msg}")

# ===================== 环境变量 =====================
log("========== 程序启动 ==========")

try:
    PUSHDEER_TOKEN = os.getenv("PUSHDEER_TOKEN")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    STOCK_LIST_STR = os.getenv("STOCK_LIST", "")

    log(f"PUSHDEER_TOKEN 存在: {PUSHDEER_TOKEN is not None}")
    log(f"DEEPSEEK_API_KEY 存在: {DEEPSEEK_API_KEY is not None}")
    log(f"股票列表: [{STOCK_LIST_STR}]")
except Exception as e:
    error(f"环境变量错误: {e}")
    sys.exit(1)

# ===================== 分割股票 =====================
STOCK_CODES = [s.strip() for s in STOCK_LIST_STR.split("-") if s.strip()]
log(f"最终股票列表: {STOCK_CODES}")

# ===================== 【稳定】获取日线收盘价（用于正确计算指标） =====================
def get_daily_closes(code, count=60):
    log(f"获取日线数据: {code}")
    try:
        url = f"https://data.gtimg.cn/flashdata/hushen/daily/{code[:2]}/{code}.js"
        resp = requests.get(url, timeout=15)
        txt = resp.text
        data_part = txt.split('"')[1]
        lines = data_part.split("\\n")
        
        closes = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    closes.append(float(parts[3]))
                except:
                    continue
        closes = closes[-count:]
        log(f"获取日线数量: {len(closes)}")
        return closes
    except Exception as e:
        error(f"获取日线失败: {e}")
        return []

# ===================== 【标准正确】RSI 指标公式 =====================
def rsi(prices, period):
    if len(prices) < period + 1:
        return 50.0
    
    deltas = np.diff(prices)
    gains = deltas.copy()
    losses = deltas.copy()
    
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    # 初始平均值
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # 滚动计算（标准公式）
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)

# ===================== 【标准】EMA20 =====================
def ema(prices, period=20):
    if len(prices) < period:
        return 0.0
    return round(pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1], 2)

# ===================== 实时名称价格 =====================
def get_basic(code):
    try:
        url = f"https://qt.gtimg.cn/q={code}"
        arr = requests.get(url, timeout=8).text.split("~")
        name = arr[1]
        price = float(arr[3])
        pre = float(arr[4])
        pct = round((price - pre) / pre * 100, 2)
        return {"name": name, "price": price, "pct": pct}
    except:
        return {"name": code, "price": 0.0, "pct": 0.0}

# ===================== AI 分析 =====================
def ai_analyze(stock):
    prompt = f"""请以资深股票技术分析师身份，对【{stock['name']} / {stock['code']}】进行技术面分析，重点聚焦价格位置、K 线形态、量价关系、支撑压力、均线趋势，简要结合行业与政策环境，语言专业精炼，不构成投资建议。

分析框架：
股价当前所处区间（高位/中位/低位），关键支撑位与压力位
K 线形态、趋势结构、均线系统表现
成交量配合情况
行业景气度及相关政策简要影响
技术面观点 + 风险提示

当前数据：
价格：{stock['price']} 元
涨跌幅：{stock['pct']}%
RSI6={stock['rsi6']}，RSI12={stock['rsi12']}，RSI24={stock['rsi24']}
EMA20={stock['ema20']}
"""
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        return client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        ).choices[0].message.content
    except Exception as e:
        return f"AI分析失败: {str(e)}"

# ===================== 推送1只股票 =====================
def push_one(stock):
    log(f"生成推送消息: {stock['code']}")
    
    msg = f"📈 个股技术指标 {datetime.date.today()}\n\n"
    msg += f"【{stock['name']}】{stock['code']}\n"
    msg += f"现价：{stock['price']} 元  {stock['pct']:+}%\n\n"
    msg += f"RSI(6)  = {stock['rsi6']}\n"
    msg += f"RSI(12) = {stock['rsi12']}\n"
    msg += f"RSI(24) = {stock['rsi24']}\n"
    msg += f"EMA(20) = {stock['ema20']}\n\n"
    msg += f"🤖 技术分析：\n{ai_analyze(stock)}"
    
    # 回显推送内容
    print("\n="*50)
    print("【PushDeer 推送消息】")
    print("="*50)
    print(msg)
    print("="*50)
    
    try:
        import urllib.parse
        push_url = f"https://api2.pushdeer.com/message/push?pushkey={PUSHDEER_TOKEN}&text={urllib.parse.quote(msg)}"
        requests.get(push_url, timeout=10)
        success(f"{stock['code']} 推送成功")
    except Exception as e:
        error(f"推送失败: {e}")

# ===================== 主程序 =====================
if __name__ == "__main__":
    import pandas as pd
    for code in STOCK_CODES:
        basic = get_basic(code)
        closes = get_daily_closes(code, count=60)
        
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
        
        push_one(stock)
    
    log("========== 全部完成 ==========")
