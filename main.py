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

# ===================== 【修复】K线数据：过滤非数字行 =====================
def get_price_data(code):
    log(f"获取K线: {code}")
    try:
        url = f"https://data.gtimg.cn/flashdata/hushen/minute/{code}.js"
        resp = requests.get(url, timeout=15)
        log(f"接口状态码: {resp.status_code}")

        data_part = resp.text.split('"')[1]
        lines = data_part.split("\\n")

        closes = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 【关键修复】跳过包含 date/字母 的行，只留纯数字K线
            if "date" in line or any(c.isalpha() for c in line):
                continue

            parts = line.split()
            if len(parts) >= 2:
                try:
                    close_val = float(parts[1])
                    closes.append(close_val)
                except:
                    continue

        log(f"有效K线数量: {len(closes)}")
        return closes[-60:] if len(closes) >= 30 else closes

    except Exception as e:
        error(f"K线获取失败: {type(e).__name__}: {e}")
        return []

# ===================== 实时价格名称 =====================
def get_basic(code):
    try:
        url = f"https://qt.gtimg.cn/q={code}"
        arr = requests.get(url, timeout=8).text.split("~")
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
    ag, al = np.mean(gains[-n:]), np.mean(losses[-n:])
    return round(100 - 100/(1 + ag/al), 2) if al !=0 else 100.0

def calc_ema(prices, n=20):
    if len(prices) < n: return 0.0
    val = prices[-1]
    a = 2/(n+1)
    for p in reversed(prices[:-1]):
        val = a*p + (1-a)*val
    return round(val,2)

# ===================== AI 分析 =====================
def ai_analyze(stock):
    prompt = f"""请以资深股票技术分析师身份，对【{stock['name']} / {stock['code']}】进行技术面分析，重点聚焦价格位置、K 线形态、量价关系、支撑压力、均线趋势，简要结合行业与政策环境，语言专业精炼，不构成投资建议。
分析框架：股价区间、支撑压力、K线与均线、成交量、行业政策、技术观点+风险
数据：价格{stock['price']}元，涨跌幅{stock['pct']}%，RSI6={stock['rsi6']}，RSI12={stock['rsi12']}，RSI24={stock['rsi24']}，EMA20={stock['ema20']}"""
    try:
        c = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        return c.chat.completions.create(model="deepseek-chat", messages=[{"role":"user","content":prompt}], temperature=0.3).choices[0].message.content
    except:
        return "AI 分析暂时不可用"

# ===================== 推送 + 消息回显 =====================
def push_one(stock):
    log("="*50)
    log(f"准备推送：{stock['code']}")

    msg = f"📈 个股技术指标 {datetime.date.today()}\n\n"
    msg += f"【{stock['name']}】{stock['code']}\n"
    msg += f"现价：{stock['price']} 元  {stock['pct']:+}%\n\n"
    msg += f"RSI(6)  = {stock['rsi6']}\n"
    msg += f"RSI(12) = {stock['rsi12']}\n"
    msg += f"RSI(24) = {stock['rsi24']}\n"
    msg += f"EMA(20) = {stock['ema20']}\n\n"
    msg += f"🤖 技术分析：\n{ai_analyze(stock)}"

    # 回显推送内容
    print("\n【PushDeer 推送消息全文】:\n")
    print(msg)
    print("\n" + "="*50 + "\n")

    # 发送
    try:
        import urllib.parse
        u = f"https://api2.pushdeer.com/message/push?pushkey={PUSHDEER_TOKEN}&text={urllib.parse.quote(msg)}"
        requests.get(u, timeout=10)
        success(f"{stock['code']} 推送成功")
    except Exception as e:
        error(f"推送失败: {e}")

# ===================== 主程序 =====================
if __name__ == "__main__":
    for code in STOCK_CODES:
        basic = get_basic(code)
        prices = get_price_data(code)

        if not prices:
            error(f"{code} 无K线数据，跳过")
            continue

        stock = {
            "code": code,
            "name": basic["name"],
            "price": round(basic["price"],2),
            "pct": round((basic["price"]-basic["pre"])/basic["pre"]*100,2) if basic["pre"] else 0,
            "rsi6": calc_rsi(prices,6),
            "rsi12": calc_rsi(prices,12),
            "rsi24": calc_rsi(prices,24),
            "ema20": calc_ema(prices,20)
        }

        push_one(stock)

    log("========== 全部完成 ==========")
