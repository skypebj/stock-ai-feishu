import os
import sys
import requests
import datetime
import numpy as np
from openai import OpenAI

# ===================== 日志 =====================
def log(msg):
    print(f"[LOG] {msg}")

def err(msg):
    print(f"[ERROR] {msg}")

def ok(msg):
    print(f"[SUCCESS] {msg}")

# ===================== 读取配置 =====================
log("========== 程序启动 ==========")

try:
    PUSHDEER_TOKEN = os.getenv("PUSHDEER_TOKEN")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    STOCK_STR = os.getenv("STOCK_LIST", "")

    log(f"PushDeer 配置: {PUSHDEER_TOKEN is not None}")
    log(f"DeepSeek 配置: {DEEPSEEK_API_KEY is not None}")
    log(f"股票原始字符串: {STOCK_STR}")
except Exception as e:
    err(f"读取环境变量失败: {e}")
    sys.exit(1)

# 按 - 分割股票
try:
    STOCKS = [s.strip() for s in STOCK_STR.split("-") if s.strip()]
    log(f"最终股票列表: {STOCKS}")
except Exception as e:
    err(f"分割股票失败: {e}")
    sys.exit(1)

# ===================== 从 data.gtimg.cn 获取K线 =====================
def get_kline(code):
    log(f"获取K线: {code}")
    try:
        url = f"https://data.gtimg.cn/flashdata/hushen/minute/{code}.js"
        resp = requests.get(url, timeout=12)
        txt = resp.text
        data = txt.split('"')[1]
        lines = data.split("\\n")
        closes = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                closes.append(float(parts[1]))
        # 取最近60根
        res = closes[-60:] if len(closes) >= 60 else closes
        ok(f"{code} K线数量: {len(res)}")
        return res
    except Exception as e:
        err(f"{code} K线获取失败: {e}")
        return []

# ===================== 实时价格 & 名称 =====================
def get_rt(code):
    try:
        url = f"https://qt.gtimg.cn/q={code}"
        r = requests.get(url, timeout=8)
        arr = r.text.split("~")
        name = arr[1]
        price = float(arr[3])
        pre = float(arr[4])
        pct = round((price - pre) / pre * 100, 2)
        return {"name": name, "price": price, "pct": pct}
    except:
        return {"name": code, "price": 0, "pct": 0}

# ===================== RSI 计算 =====================
def calc_rsi(prices, n):
    if len(prices) < n + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_g = np.mean(gains[-n:])
    avg_l = np.mean(losses[-n:])
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return round(100 - (100 / (1 + rs)), 2)

# ===================== EMA20 计算 =====================
def calc_ema(prices, n=20):
    if len(prices) < n:
        return 0.0
    ema_val = prices[-1]
    alpha = 2 / (n + 1)
    for p in reversed(prices[:-1]):
        ema_val = alpha * p + (1 - alpha) * ema_val
    return round(ema_val, 2)

# ===================== 获取一只股票的完整数据 =====================
def get_one(code):
    rt = get_rt(code)
    kl = get_kline(code)
    if not kl:
        return None
    return {
        "code": code,
        "name": rt["name"],
        "price": rt["price"],
        "pct": rt["pct"],
        "rsi6": calc_rsi(kl, 6),
        "rsi12": calc_rsi(kl, 12),
        "rsi24": calc_rsi(kl, 24),
        "ema20": calc_ema(kl, 20)
    }

# ===================== AI 分析（你给的完整提示词） =====================
def ai_ana(stock):
    log(f"AI 分析: {stock['code']}")
    prompt = f"""请以资深股票技术分析师身份，对【{stock['name']} / {stock['code']}】进行技术面分析，重点聚焦价格位置、K 线形态、量价关系、支撑压力、均线趋势，简要结合行业与政策环境，语言专业精炼，不构成投资建议。

分析框架：
股价当前所处区间（高位 / 中位 / 低位），关键支撑位与压力位
K 线形态、趋势结构、均线系统表现
成交量配合情况
行业景气度及相关政策简要影响
技术面观点 + 风险提示

当前数据：
价格：{stock['price']} 元，涨跌幅：{stock['pct']}%
RSI6={stock['rsi6']}，RSI12={stock['rsi12']}，RSI24={stock['rsi24']}
EMA20={stock['ema20']}
"""
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI分析失败: {str(e)}"

# ===================== PushDeer 推送 =====================
def push(text):
    try:
        import urllib.parse
        enc = urllib.parse.quote(text)
        url = f"https://api2.pushdeer.com/message/push?pushkey={PUSHDEER_TOKEN}&text={enc}"
        requests.get(url, timeout=10)
        ok("PushDeer 推送成功")
    except Exception as e:
        err(f"PushDeer 推送失败: {e}")

# ===================== 主程序 =====================
if __name__ == "__main__":
    log("开始获取股票数据")
    data_list = []
    for code in STOCKS:
        d = get_one(code)
        if d:
            data_list.append(d)

    if not data_list:
        err("无有效股票数据")
        sys.exit(1)

    # 组装消息
    msg = f"📈 技术指标推送 {datetime.date.today()}\n\n"
    for s in data_list:
        msg += f"【{s['name']}】{s['code']}\n"
        msg += f"现价：{s['price']} 元  {s['pct']}%\n"
        msg += f"RSI(6)={s['rsi6']}  RSI(12)={s['rsi12']}  RSI(24)={s['rsi24']}\n"
        msg += f"EMA(20)={s['ema20']}\n\n"
        msg += f"🤖 AI 分析：\n{ai_ana(s)}\n\n"
        msg += "----------------------------------------\n\n"

    push(msg)
    log("========== 全部执行完成 ==========")
