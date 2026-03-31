import os
import sys
import requests
import datetime
import numpy as np
from openai import OpenAI

# ===================== 超级详细 DEBUG 日志 =====================
def log(msg):
    print(f"[DEBUG] {msg}")

def warn(msg):
    print(f"[WARN] {msg}")

def error(msg):
    print(f"[ERROR] {msg}")

def success(msg):
    print(f"[SUCCESS] {msg}")

# ===================== 读取环境变量 =====================
log("========== 程序启动 ==========")

try:
    PUSHDEER_TOKEN = os.getenv("PUSHDEER_TOKEN")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    STOCK_LIST_STR = os.getenv("STOCK_LIST", "")

    log(f"PUSHDEER_TOKEN 存在: {PUSHDEER_TOKEN is not None}")
    log(f"DEEPSEEK_API_KEY 存在: {DEEPSEEK_API_KEY is not None}")
    log(f"STOCK_LIST_STR: [{STOCK_LIST_STR}]")

except Exception as e:
    error(f"读取环境变量异常: {str(e)}")
    sys.exit(1)

# ===================== 分割股票列表 =====================
try:
    STOCK_CODES = [x.strip() for x in STOCK_LIST_STR.split("-") if x.strip()]
    log(f"分割后股票列表: {STOCK_CODES}")
except Exception as e:
    error(f"分割股票失败: {str(e)}")
    sys.exit(1)

# ===================== 获取K线数据（超细日志） =====================
def get_price_data(code):
    log(f"→ 处理股票: [{code}]")
    try:
        url = f"https://data.gtimg.cn/flashdata/hushen/minute/{code}.js"
        log(f"请求URL: {url}")

        resp = requests.get(url, timeout=15)
        log(f"HTTP状态码: {resp.status_code}")

        if resp.status_code != 200:
            error(f"请求失败 {resp.status_code}")
            return []

        data_part = resp.text.split('"')[1]
        lines = data_part.split("\\n")

        closes = []
        for line in lines:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 2:
                closes.append(float(parts[1]))

        log(f"获取K线数量: {len(closes)}")
        return closes[-60:] if len(closes) >= 60 else closes

    except Exception as e:
        error(f"获取K线异常: {type(e).__name__}: {str(e)}")
        return []

# ===================== 实时名称价格 =====================
def get_stock_basic(code):
    try:
        url = f"https://qt.gtimg.cn/q={code}"
        r = requests.get(url, timeout=8)
        arr = r.text.split("~")
        return {
            "name": arr[1] if len(arr) > 1 else code,
            "price": float(arr[3]) if len(arr) > 3 else 0.0,
            "pre": float(arr[4]) if len(arr) > 4 else 0.0
        }
    except:
        return {"name": code, "price": 0.0, "pre": 0.0}

# ===================== RSI / EMA 计算 =====================
def calc_rsi(prices, n):
    if len(prices) < n+1: return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_g = np.mean(gains[-n:])
    avg_l = np.mean(losses[-n:])
    if avg_l == 0: return 100.0
    rs = avg_g / avg_l
    return round(100 - 100/(1+rs), 2)

def calc_ema(prices, n=20):
    if len(prices) < n: return 0.0
    ema_val = prices[-1]
    alpha = 2/(n+1)
    for p in reversed(prices[:-1]):
        ema_val = alpha * p + (1-alpha)*ema_val
    return round(ema_val, 2)

# ===================== AI 分析（你的完整提示词） =====================
def ai_analysis(stock):
    log(f"AI 分析: {stock['code']}")
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
RSI6={stock['rsi6']} RSI12={stock['rsi12']} RSI24={stock['rsi24']}
EMA20={stock['ema20']}
"""
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        res = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"AI分析失败: {str(e)}"

# ===================== 单只股票推送 1 条 PushDeer =====================
def push_single(stock):
    log(f"开始推送 {stock['code']}")
    msg = f"📈 个股技术指标 {datetime.date.today()}\n\n"
    msg += f"【{stock['name']}】{stock['code']}\n"
    msg += f"现价：{stock['price']} 元  {stock['pct']:+}%\n\n"
    msg += f"RSI(6)={stock['rsi6']}\n"
    msg += f"RSI(12)={stock['rsi12']}\n"
    msg += f"RSI(24)={stock['rsi24']}\n"
    msg += f"EMA(20)={stock['ema20']}\n\n"
    msg += f"🤖 技术分析：\n{ai_analysis(stock)}"

    try:
        import urllib.parse
        enc = urllib.parse.quote(msg)
        url = f"https://api2.pushdeer.com/message/push?pushkey={PUSHDEER_TOKEN}&text={enc}"
        requests.get(url, timeout=10)
        success(f"{stock['code']} 推送成功")
    except Exception as e:
        error(f"{stock['code']} 推送失败: {str(e)}")

# ===================== 主逻辑 =====================
if __name__ == "__main__":
    for code in STOCK_CODES:
        basic = get_stock_basic(code)
        prices = get_price_data(code)
        if not prices:
            error(f"{code} 无K线数据，跳过")
            continue

        stock = {
            "code": code,
            "name": basic["name"],
            "price": round(basic["price"],2),
            "pct": round((basic["price"]-basic["pre"])/basic["pre"]*100,2) if basic["pre"]!=0 else 0,
            "rsi6": calc_rsi(prices,6),
            "rsi12": calc_rsi(prices,12),
            "rsi24": calc_rsi(prices,24),
            "ema20": calc_ema(prices,20)
        }

        # 每只股票单独推一次
        push_single(stock)

    log("========== 全部执行完成 ==========")
