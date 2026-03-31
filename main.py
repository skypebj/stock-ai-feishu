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

# ===================== 读取环境变量（打印每一步） =====================
log("========== 程序启动 ==========")

# 读取 3 个密钥
try:
    PUSHDEER_TOKEN = os.getenv("PUSHDEER_TOKEN")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    STOCK_LIST_STR = os.getenv("STOCK_LIST", "")

    log(f"PUSHDEER_TOKEN 读取成功: {PUSHDEER_TOKEN is not None}")
    log(f"DEEPSEEK_API_KEY 读取成功: {DEEPSEEK_API_KEY is not None}")
    log(f"STOCK_LIST_STR 原始内容: [{STOCK_LIST_STR}]")

except Exception as e:
    error(f"环境变量读取异常: {str(e)}")
    sys.exit(1)

# 分割股票（打印每一步）
try:
    if not STOCK_LIST_STR:
        error("股票列表为空！请检查 GitHub Secret STOCK_LIST")
        sys.exit(1)

    STOCK_CODES = [x.strip() for x in STOCK_LIST_STR.split("-") if x.strip()]
    log(f"按 '-' 分割后股票列表: {STOCK_CODES}")
    log(f"待获取股票数量: {len(STOCK_CODES)}")

except Exception as e:
    error(f"股票分割失败: {str(e)}")
    sys.exit(1)

# ===================== 【关键】获取K线数据（打印所有断点） =====================
def get_price_data(code):
    log(f"→ 开始获取股票: [{code}]")

    # 断点1：检查代码格式
    if not code.startswith("sh") and not code.startswith("sz"):
        warn(f"股票代码格式异常，不是 sh/sz 开头: {code}")
        return []

    try:
        # 断点2：打印请求URL
        url = f"https://data.gtimg.cn/flashdata/hushen/minute/{code}.js"
        log(f"请求URL: {url}")

        resp = requests.get(url, timeout=15)
        log(f"HTTP 状态码: {resp.status_code}")
        log(f"返回内容长度: {len(resp.text)}")

        if resp.status_code != 200:
            error(f"接口请求失败，状态码: {resp.status_code}")
            return []

        # 断点3：打印原始内容片段
        snippet = resp.text[:100].replace("\n", " ")
        log(f"接口返回片段: {snippet}...")

        # 断点4：尝试分割数据
        if '"' not in resp.text:
            error("返回内容不包含双引号，格式错误！")
            return []

        data_part = resp.text.split('"')[1]
        log(f"数据段提取成功，长度: {len(data_part)}")

        # 断点5：按行拆分
        lines = data_part.split("\\n")
        log(f"K线行数: {len(lines)}")

        # 提取收盘价
        closes = []
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            # 断点6：打印每行结构
            if idx < 3:
                log(f"行{idx} 内容: {line} | 分割后: {parts}")

            if len(parts) >= 2:
                try:
                    close_val = float(parts[1])
                    closes.append(close_val)
                except:
                    continue

        log(f"有效收盘价数量: {len(closes)}")
        return closes[-60:] if len(closes) >= 60 else closes

    except Exception as e:
        error(f"获取K线崩溃: {type(e).__name__}: {str(e)}")
        return []

# ===================== 实时名称 & 价格 =====================
def get_stock_basic(code):
    try:
        url = f"https://qt.gtimg.cn/q={code}"
        r = requests.get(url, timeout=8)
        arr = r.text.split("~")
        return {
            "name": arr[1] if len(arr) > 1 else code,
            "price": float(arr[3]) if len(arr) > 3 else 0,
            "pre": float(arr[4]) if len(arr) > 4 else 0
        }
    except:
        return {"name": code, "price": 0, "pre": 0}

# ===================== RSI & EMA 计算 =====================
def calculate_rsi(prices, n):
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

def calculate_ema(prices, n=20):
    if len(prices) < n:
        return 0.0
    ema_val = prices[-1]
    alpha = 2 / (n + 1)
    for p in reversed(prices[:-1]):
        ema_val = alpha * p + (1 - alpha) * ema_val
    return round(ema_val, 2)

# ===================== 单只股票完整获取 =====================
def get_stock_full(code):
    log(f"\n===== 处理股票: {code} =====")
    basic = get_stock_basic(code)
    prices = get_price_data(code)

    if not prices:
        error(f"{code} 无K线数据，跳过")
        return None

    return {
        "code": code,
        "name": basic["name"],
        "price": round(basic["price"], 2),
        "pct": round((basic["price"] - basic["pre"]) / basic["pre"] * 100, 2) if basic["pre"] != 0 else 0,
        "rsi6": calculate_rsi(prices, 6),
        "rsi12": calculate_rsi(prices, 12),
        "rsi24": calculate_rsi(prices, 24),
        "ema20": calculate_ema(prices, 20)
    }

# ===================== AI 分析（你给的完整提示词） =====================
def ai_analysis(stock):
    log(f"开始AI分析: {stock['code']}")
    prompt = f"""请以资深股票技术分析师身份，对【{stock['name']} / {stock['code']}】进行技术面分析，重点聚焦价格位置、K 线形态、量价关系、支撑压力、均线趋势，简要结合行业与政策环境，语言专业精炼，不构成投资建议。
分析框架：
股价当前所处区间（高位 / 中位 / 低位），关键支撑位与压力位
K 线形态、趋势结构、均线系统表现
成交量配合情况
行业景气度及相关政策简要影响
技术面观点 + 风险提示

当前数据：价格={stock['price']}元, 涨跌幅={stock['pct']}%, RSI6={stock['rsi6']}, RSI12={stock['rsi12']}, RSI24={stock['rsi24']}, EMA20={stock['ema20']}
"""
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        res = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.3)
        return res.choices[0].message.content
    except Exception as e:
        return f"AI分析失败: {str(e)}"

# ===================== PushDeer 推送 =====================
def push_to_deer(text):
    try:
        import urllib.parse
        payload = urllib.parse.quote(text)
        url = f"https://api2.pushdeer.com/message/push?pushkey={PUSHDEER_TOKEN}&text={payload}"
        requests.get(url, timeout=10)
        success("PushDeer 推送成功")
    except Exception as e:
        error(f"推送失败: {str(e)}")

# ===================== 主入口 =====================
if __name__ == "__main__":
    log("开始执行主任务")
    stock_results = []

    for code in STOCK_CODES:
        data = get_stock_full(code)
        if data:
            stock_results.append(data)
            success(f"{code} 处理完成 ✅")

    if not stock_results:
        error("无任何股票数据")
        sys.exit(1)

    # 组装消息
    msg = f"📈 A股技术指标推送 {datetime.date.today()}\n\n"
    for s in stock_results:
        msg += f"【{s['name']}】{s['code']}\n"
        msg += f"现价：{s['price']} 元  {s['pct']}%\n"
        msg += f"RSI(6)={s['rsi6']}  RSI(12)={s['rsi12']}  RSI(24)={s['rsi24']}\n"
        msg += f"EMA(20)={s['ema20']}\n\n"
        msg += f"🤖 AI分析:\n{ai_analysis(s)}\n\n"
        msg += "-" * 50 + "\n\n"

    push_to_deer(msg)
    log("========== 程序全部执行完成 ==========")
