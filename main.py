import os
import datetime
import requests
import numpy as np
import pandas as pd
import baostock as bs

# ===================== 调试日志 =====================
def log(msg): print(f"\033[34m[DEBUG]\033[0m {msg}")
def error(msg): print(f"\033[31m[ERROR]\033[0m {msg}")
def var(name, value): print(f"\033[33m[VAR]\033[0m {name} = {value}")
def step(msg): print(f"\n\033[32m==================== {msg} ====================\033[0m")

# ===================== 环境变量 =====================
step("启动程序")
PUSHDEER_TOKEN = os.getenv("PUSHDEER_TOKEN")
STOCK_LIST_STR = os.getenv("STOCK_LIST", "sh513100-sh601328")

var("原始股票字符串", STOCK_LIST_STR)
var("PUSHDEER 是否配置", PUSHDEER_TOKEN is not None)

STOCK_CODES = [s.strip() for s in STOCK_LIST_STR.split("-") if s.strip()]
var("最终股票列表", STOCK_CODES)

# ===================== Baostock 登录（全局一次） =====================
def bs_login():
    step("Baostock 登录")
    lg = bs.login()
    var("登录码", lg.error_code)
    var("登录信息", lg.error_msg)
    if lg.error_code != "0":
        error("Baostock 登录失败")
        return False
    return True

# ===================== 【核心】Baostock 获取日线（超级调试） =====================
def get_bs_daily_debug(code):
    step(f"Baostock 获取 {code} 日线")
    try:
        # 转换 Baostock 代码格式：sh513100 → sh.513100
        bs_code = f"{code[:2]}.{code[2:]}"
        var("Baostock 代码", bs_code)

        # 取最近 60 个交易日（可调整）
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
        var("开始日期", start_date)
        var("结束日期", end_date)

        # 查询日线（前复权）
        rs = bs.query_history_k_data_plus(
            code=bs_code,
            fields="date,close",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"  # 3=前复权
        )

        var("查询错误码", rs.error_code)
        var("查询错误信息", rs.error_msg)
        if rs.error_code != "0":
            error("查询失败")
            return []

        # 转 DataFrame
        df = rs.get_data()
        var("返回数据形状", df.shape)
        var("返回列名", list(df.columns))
        if df.empty:
            error("返回空数据")
            return []

        # 清洗收盘价
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        closes = df["close"].tolist()[-60:]  # 取最近 60 条

        var("有效收盘价数量", len(closes))
        var("最近5日收盘价", [round(x,3) for x in closes[-5:]])
        return closes

    except Exception as e:
        error(f"异常：{str(e)}")
        import traceback
        traceback.print_exc()
        return []

# ===================== 实时行情（腾讯，稳定） =====================
def get_quote_debug(code):
    step(f"获取实时行情 {code}")
    try:
        url = f"https://qt.gtimg.cn/q={code}"
        var("请求URL", url)
        txt = requests.get(url, timeout=8).text
        var("响应长度", len(txt))
        arr = txt.split("~")
        var("分割字段数", len(arr))

        name = arr[1] if len(arr) > 1 else code
        price = float(arr[3]) if len(arr) > 3 else 0.0
        pre_close = float(arr[4]) if len(arr) > 4 else price
        pct = round((price - pre_close) / pre_close * 100, 2) if pre_close != 0 else 0

        var("名称", name)
        var("现价", price)
        var("涨跌幅", pct)
        return {"name": name, "price": price, "pct": pct}
    except Exception as e:
        error(f"行情异常：{e}")
        return {"name": code, "price": 0, "pct": 0}

# ===================== RSI 计算 =====================
def rsi_debug(prices, n):
    step(f"计算 RSI({n})")
    var("输入K线数量", len(prices))
    if len(prices) < n + 1:
        var("数据不足，返回50", True)
        return 50.0

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_g = np.mean(gains[:n])
    avg_l = np.mean(losses[:n])
    for i in range(n, len(gains)):
        avg_g = (avg_g * (n-1) + gains[i]) / n
        avg_l = (avg_l * (n-1) + losses[i]) / n

    rs = avg_g / avg_l if avg_l != 0 else 999
    res = 100 - 100 / (1 + rs)
    var("RSI结果", round(res, 2))
    return round(res, 2)

# ===================== EMA 计算 =====================
def ema_debug(prices, n=20):
    step(f"计算 EMA({n})")
    var("K线数量", len(prices))
    if len(prices) < n:
        var("数据不足，返回0", True)
        return 0.0
    ema_val = round(pd.Series(prices).ewm(span=n, adjust=False).mean().iloc[-1], 2)
    var("EMA结果", ema_val)
    return ema_val

# ===================== 推送 =====================
def push_debug(stock):
    step(f"推送 {stock['code']}")
    msg = (
        f"📈 {datetime.date.today()}\n"
        f"【{stock['name']}】{stock['code']}\n"
        f"现价：{stock['price']} {stock['pct']:+}%\n"
        f"RSI6={stock['rsi6']} RSI12={stock['rsi12']} RSI24={stock['rsi24']}\n"
        f"EMA20={stock['ema20']}"
    )
    var("推送消息", msg)
    if PUSHDEER_TOKEN:
        requests.get(f"https://api2.pushdeer.com/message/push?pushkey={PUSHDEER_TOKEN}&text={requests.utils.quote(msg)}")

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 登录 Baostock
    if not bs_login():
        error("程序终止：Baostock 登录失败")
        exit(1)

    for code in STOCK_CODES:
        step(f"处理股票：{code}")
        basic = get_quote_debug(code)
        closes = get_bs_daily_debug(code)

        if len(closes) < 20:
            error(f"{code} 数据不足，跳过")
            continue

        stock = {
            "code": code,
            "name": basic["name"],
            "price": basic["price"],
            "pct": basic["pct"],
            "rsi6": rsi_debug(closes, 6),
            "rsi12": rsi_debug(closes, 12),
            "rsi24": rsi_debug(closes, 24),
            "ema20": ema_debug(closes, 20)
        }

        var("最终股票对象", stock)
        push_debug(stock)

    # 登出 Baostock
    bs.logout()
    step("全部执行完成 ✅")
