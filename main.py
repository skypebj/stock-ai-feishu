import os
import sys
import requests
import datetime
import pandas as pd
import numpy as np
from openai import OpenAI

# ===================== 日志输出 =====================
def log(message):
    print(f"[LOG] {message}")

def error(message):
    print(f"[ERROR] {message}")

def success(message):
    print(f"[SUCCESS] {message}")

# ========================================================
log("========== 程序启动 ==========")

# 读取环境变量
try:
    PUSHDEER_TOKEN = os.getenv("PUSHDEER_TOKEN")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    STOCK_LIST_STR = os.getenv("STOCK_LIST", "")
    
    log(f"PushDeer Token 已读取: {PUSHDEER_TOKEN is not None}")
    log(f"DeepSeek Key 已读取: {DEEPSEEK_API_KEY is not None}")
except Exception as e:
    error(f"读取环境变量失败: {e}")
    sys.exit(1)

# 分割股票列表
try:
    STOCK_LIST = [s.strip() for s in STOCK_LIST_STR.split("-") if s.strip()]
    log(f"分割后股票列表: {STOCK_LIST}")
except Exception as e:
    error(f"分割股票失败: {e}")
    sys.exit(1)

# ===================== 从 data.gtimg.cn 获取日K数据 =====================
def get_daily_k(code):
    """
    从 data.gtimg.cn 获取日K线（前复权）
    返回: DataFrame[date, open, high, low, close, volume]
    """
    log(f"开始获取 {code} 日K数据")
    try:
        # 转换为接口格式：sh601328 / sz513100
        code_low = code.lower()
        market = code_low[:2]
        symbol = code_low[2:]
        url = f"https://data.gtimg.cn/flashdata/hushen/daily/{market}/{code_low}.js"
        
        resp = requests.get(url, timeout=15)
        txt = resp.text.strip()
        # 提取数据部分
        data_str = txt.split('="')[1].split('";')[0]
        # 按换行分割
        lines = data_str.split('\n')
        data = []
        for line in lines:
            if not line:
                continue
            # 格式: year:'2026',month:'3',day:'31',open:'5.67',high:'5.69',low:'5.65',close:'5.67',volume:'12345678'
            items = line.replace("'", "").split(',')
            row = {}
            for item in items:
                k, v = item.split(':')
                row[k] = v
            # 构造日期
            date = f"{row['year']}-{row['month'].zfill(2)}-{row['day'].zfill(2)}"
            data.append({
                "date": date,
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume'])
            })
        df = pd.DataFrame(data)
        df = df.sort_values("date").reset_index(drop=True)
        success(f"{code} 日K获取成功，共 {len(df)} 条")
        return df
    except Exception as e:
        error(f"{code} 日K获取失败: {str(e)}")
        return pd.DataFrame()

# ===================== 计算 RSI(6,12,24) =====================
def calc_rsi(close_series, window):
    """计算单周期RSI"""
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    # 避免除零
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rs = rs.fillna(0)
    rsi = 100 - (100 / (1 + rs))
    return rsi.round(2)

# ===================== 计算 EMA(20) =====================
def calc_ema(close_series, window=20):
    """计算EMA"""
    return close_series.ewm(span=window, adjust=False).mean().round(2)

# ===================== 获取股票完整数据（价格+指标） =====================
def get_stock_data(code):
    log(f"开始处理 {code}")
    df = get_daily_k(code)
    if df.empty:
        return None
    
    # 计算指标
    close = df["close"]
    df["rsi6"] = calc_rsi(close, 6)
    df["rsi12"] = calc_rsi(close, 12)
    df["rsi24"] = calc_rsi(close, 24)
    df["ema20"] = calc_ema(close, 20)
    
    # 取最新数据
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df)>=2 else latest
    
    result = {
        "code": code,
        "name": code,  # 可从实时接口补充，此处简化
        "date": latest["date"],
        "close": latest["close"],
        "rsi6": latest["rsi6"],
        "rsi12": latest["rsi12"],
        "rsi24": latest["rsi24"],
        "ema20": latest["ema20"],
        "prev_close": prev["close"],
        "pct": round((latest["close"] - prev["close"]) / prev["close"] * 100, 2)
    }
    success(f"{code} 指标计算完成")
    return result

# ===================== AI 技术面分析（按你给的提示词） =====================
def ai_tech_analysis(stock):
    log(f"开始AI分析 {stock['code']}")
    if not DEEPSEEK_API_KEY:
        error("DeepSeek API Key 为空")
        return "未配置 DeepSeek Key"
    
    prompt = f"""请以资深股票技术分析师身份，对【{stock['name']} / {stock['code']}】进行技术面分析，重点聚焦价格位置、K线形态、量价关系、支撑压力、均线趋势，简要结合行业与政策环境，语言专业精炼，不构成投资建议。
分析框架：
1. 股价当前所处区间（高位/中位/低位），关键支撑位与压力位
2. K线形态、趋势结构、均线系统表现（EMA20={stock['ema20']}）
3. 成交量配合情况
4. 行业景气度及相关政策简要影响
5. 技术面观点 + 风险提示

当前数据：
收盘价：{stock['close']} 元，涨跌幅：{stock['pct']}%
RSI(6)={stock['rsi6']}，RSI(12)={stock['rsi12']}，RSI(24)={stock['rsi24']}
EMA(20)={stock['ema20']}
"""
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800
        )
        success(f"{stock['code']} AI分析完成")
        return response.choices[0].message.content
    except Exception as e:
        error(f"AI分析失败: {e}")
        return f"AI分析异常: {str(e)}"

# ===================== PushDeer 推送 =====================
def push_pushdeer(text):
    log("开始 PushDeer 推送")
    try:
        import urllib.parse
        text_encoded = urllib.parse.quote(text)
        url = f"https://api2.pushdeer.com/message/push?pushkey={PUSHDEER_TOKEN}&text={text_encoded}"
        resp = requests.get(url, timeout=15)
        log(f"PushDeer 响应: {resp.status_code}")
        success("推送完成")
    except Exception as e:
        error(f"推送失败: {e}")

# ===================== 主流程 =====================
if __name__ == "__main__":
    log("进入主流程（每日 9:00/13:30/14:50 执行）")
    
    all_stocks = []
    for code in STOCK_LIST:
        data = get_stock_data(code)
        if data:
            all_stocks.append(data)
    
    if not all_stocks:
        error("无有效股票数据，程序退出")
        sys.exit(1)
    
    # 构造推送内容
    msg = f"📈 股票技术指标推送 {datetime.date.today()}\n\n"
    for s in all_stocks:
        msg += f"【{s['code']}】{s['date']}\n"
        msg += f"现价：{s['close']} 元 | 涨跌幅：{s['pct']}%\n"
        msg += f"RSI(6)：{s['rsi6']} | RSI(12)：{s['rsi12']} | RSI(24)：{s['rsi24']}\n"
        msg += f"EMA(20)：{s['ema20']}\n\n"
        # AI分析
        ai_result = ai_tech_analysis(s)
        msg += f"🤖 技术面分析：\n{ai_result}\n\n"
        msg += "----------------------------------------\n\n"
    
    push_pushdeer(msg)
    log("========== 程序全部执行成功 ==========")
