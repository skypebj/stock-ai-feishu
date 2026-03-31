import os
import datetime
import requests
import numpy as np
import pandas as pd
import akshare as ak
from openai import OpenAI

# ===================== 【终极调试】所有内容全输出 =====================
def log(msg): print(f"\033[34m[DEBUG]\033[0m {msg}")
def error(msg): print(f"\033[31m[ERROR]\033[0m {msg}")
def var(name, value): print(f"\033[33m[VAR]\033[0m {name} = {value}")
def step(msg): print(f"\n\033[32m==================== {msg} ====================\033[0m")

# ===================== 环境变量 =====================
step("启动程序")
PUSHDEER_TOKEN = os.getenv("PUSHDEER_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
STOCK_LIST_STR = os.getenv("STOCK_LIST", "sh513100-sh601328")

var("原始股票字符串", STOCK_LIST_STR)
var("PUSHDEER 是否配置", PUSHDEER_TOKEN is not None)
var("DEEPSEEK 是否配置", DEEPSEEK_API_KEY is not None)

STOCK_CODES = [s.strip() for s in STOCK_LIST_STR.split("-") if s.strip()]
var("最终股票列表", STOCK_CODES)

# ===================== 【超级调试】AKShare 获取日线 =====================
def get_ak_daily_debug(code):
    step(f"开始获取 {code} 日线")
    try:
        var("输入代码", code)
        pure_code = code[2:]
        prefix = code[:2]
        var("市场前缀", prefix)
        var("纯数字代码", pure_code)

        # 自动判断类型
        is_etf = pure_code.startswith("51") or pure_code.startswith("15")
        var("是否ETF", is_etf)

        df = None
        if is_etf:
            log("调用：fund_etf_hist_em")
            df = ak.fund_etf_hist_em(symbol=pure_code, period="daily", adjust="qfq")
        else:
            log("调用：stock_zh_a_hist")
            df = ak.stock_zh_a_hist(symbol=pure_code, period="daily", adjust="qfq")

        var("返回数据形状", df.shape)
        var("返回列名", list(df.columns))
        var("前5行数据", "\n" + df.head().to_string())

        if df.empty:
            error("返回空 DataFrame")
            return []

        # 统一列名
        df.columns = [str(c).strip() for c in df.columns]
        var("清理后列名", list(df.columns))

        if "收盘" not in df.columns:
            error("致命错误：不存在【收盘】列")
            return []

        closes = df["收盘"].dropna().tolist()[-60:]
        var("有效收盘价数量", len(closes))
        var("最近5日收盘价", closes[-5:])
        return closes

    except Exception as e:
        error(f"捕获异常：{str(e)}")
        import traceback
        traceback.print_exc()
        return []

# ===================== 实时行情 =====================
def get_quote_debug(code):
    step(f"获取实时行情 {code}")
    try:
        url = f"https://qt.gtimg.cn/q={code}"
        var("请求URL", url)
        txt = requests.get(url, timeout=5).text
        var("返回内容长度", len(txt))
        arr = txt.split("~")
        var("分割字段数", len(arr))
        name = arr[1] if len(arr) > 1 else code
        price = float(arr[3]) if len(arr) > 3 else 0.0
        pct = round((float(arr[3])-float(arr[4]))/float(arr[4])*100,2) if len(arr) >4 else 0
        var("名称", name)
        var("价格", price)
        var("涨跌幅", pct)
        return {"name":name,"price":price,"pct":pct}
    except Exception as e:
        error(f"行情异常：{e}")
        return {"name":code,"price":0,"pct":0}

# ===================== RSI 调试 =====================
def rsi_debug(prices, n):
    step(f"计算 RSI({n})")
    var("输入K线数量", len(prices))
    if len(prices) < n+1:
        var("数据不足，返回50", True)
        return 50.0

    deltas = np.diff(prices)
    gains = np.where(deltas>0, deltas,0)
    losses = np.where(deltas<0,-deltas,0)

    var("deltas前10", deltas[:10])
    var("gains前10", gains[:10])
    var("losses前10", losses[:10])

    avg_g = np.mean(gains[:n])
    avg_l = np.mean(losses[:n])
    var("初始avg_g", avg_g)
    var("初始avg_l", avg_l)

    for i in range(n, len(gains)):
        avg_g = (avg_g*(n-1)+gains[i])/n
        avg_l = (avg_l*(n-1)+losses[i])/n

    var("最终avg_g", avg_g)
    var("最终avg_l", avg_l)
    rs = avg_g/avg_l if avg_l !=0 else 999
    res = 100 - 100/(1+rs)
    var("RS", rs)
    var("RSI结果", round(res,2))
    return round(res,2)

# ===================== EMA 调试 =====================
def ema_debug(prices, n=20):
    step(f"计算 EMA({n})")
    var("K线数量", len(prices))
    if len(prices)<n:
        var("数据不足，返回0", True)
        return 0.0
    ema_val = round(pd.Series(prices).ewm(span=n, adjust=False).mean().iloc[-1],2)
    var("EMA结果", ema_val)
    return ema_val

# ===================== 推送 =====================
def push_debug(stock):
    step(f"推送 {stock['code']}")
    msg = f"📈 {datetime.date.today()}\n【{stock['name']}】{stock['code']}\n现价：{stock['price']} {stock['pct']:+}%\nRSI6={stock['rsi6']} RSI12={stock['rsi12']} RSI24={stock['rsi24']}\nEMA20={stock['ema20']}"
    var("推送消息", msg)
    if PUSHDEER_TOKEN:
        requests.get(f"https://api2.pushdeer.com/message/push?pushkey={PUSHDEER_TOKEN}&text={requests.utils.quote(msg)}")

# ===================== 主程序 =====================
if __name__ == "__main__":
    step("开始执行")
    for code in STOCK_CODES:
        step(f"处理股票：{code}")
        basic = get_quote_debug(code)
        closes = get_ak_daily_debug(code)

        if len(closes) < 20:
            error(f"{code} 数据不足，跳过")
            continue

        stock = {
            "code": code,
            "name": basic["name"],
            "price": basic["price"],
            "pct": basic["pct"],
            "rsi6": rsi_debug(closes,6),
            "rsi12": rsi_debug(closes,12),
            "rsi24": rsi_debug(closes,24),
            "ema20": ema_debug(closes,20)
        }

        var("最终股票对象", stock)
        push_debug(stock)

    step("全部执行完成")
