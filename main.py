import os
import datetime
import requests
import numpy as np
import pandas as pd
import baostock as bs
from openai import OpenAI

# ===================== 技术指标计算 =====================
def rsi(prices, n):
    if len(prices) < n + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g = np.mean(gains[:n])
    avg_l = np.mean(losses[:n])
    for i in range(n, len(gains)):
        avg_g = (avg_g * (n - 1) + gains[i]) / n
        avg_l = (avg_l * (n - 1) + losses[i]) / n
    rs = avg_g / avg_l if avg_l != 0 else 999
    return round(100 - 100 / (1 + rs), 2)

def ema(prices, n=20):
    if len(prices) < n:
        return 0.0
    return round(pd.Series(prices).ewm(span=n, adjust=False).mean().iloc[-1], 2)

# ===================== 行情数据获取 =====================
def get_daily(code):
    bs_code = f"{code[:2]}.{code[2:]}"
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=120)).strftime("%Y-%m-%d")
    rs = bs.query_history_k_data_plus(bs_code, "date,close", start_date, end_date, "d", "3")
    df = rs.get_data()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df["close"].dropna().tolist()[-60:]

def get_quote(code):
    try:
        url = f"https://qt.gtimg.cn/q={code}"
        txt = requests.get(url, timeout=5).text
        arr = txt.split("~")
        name = arr[1]
        price = float(arr[3])
        pre_close = float(arr[4])
        pct = round((price - pre_close) / pre_close * 100, 2)
        return {"name": name, "price": price, "pct": pct}
    except Exception as e:
        return {"name": code, "price": 0.0, "pct": 0.0}

# ===================== AI 专业分析 =====================
def ai_analyze(stock):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return "未配置 DeepSeek API Key"

    prompt = f"""请以资深股票技术分析师身份，对【{stock['name']}/{stock['code']}】进行技术面分析，重点聚焦价格位置、K 线形态、量价关系、支撑压力、均线趋势，简要结合行业与政策环境，语言专业精炼，形成投资建议。
分析框架：
股价当前所处区间（高位 / 中位 / 低位），关键支撑位与压力位
K 线形态、趋势结构、均线系统表现
成交量配合情况
行业景气度及相关政策简要影响
技术面观点 + 风险提示

当前指标：
现价：{stock['price']}
涨跌幅：{stock['pct']}%
RSI6：{stock['rsi6']}
RSI12：{stock['rsi12']}
RSI24：{stock['rsi24']}
EMA20：{stock['ema20']}
输出不超过300字"""
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"AI分析失败：{str(e)}"

# ===================== PushDeer 推送 =====================
def push_to_pushdeer(content):
    token = os.getenv("PUSHDEER_TOKEN")
    if not token:
        print("ℹ PushDeer 未配置，跳过推送")
        return
    try:
        url = "https://api2.pushdeer.com/message/push"
        params = {"pushkey": token, "text": content}
        requests.get(url, params=params, timeout=10)
        print("✅ PushDeer 推送成功")
    except Exception as e:
        print(f"❌ PushDeer 推送失败：{e}")

# ===================== Server酱 推送 =====================
def push_to_serverchan(content):
    key = os.getenv("SERVERCHAN_KEY")
    if not key:
        print("ℹ Server酱 未配置，跳过推送")
        return
    try:
        url = f"https://sctapi.ftqq.com/{key}.send"
        data = {
            "title": "📈 股票技术分析报告",
            "desp": content
        }
        requests.post(url, data=data, timeout=10)
        print("✅ Server酱 推送成功")
    except Exception as e:
        print(f"❌ Server酱 推送失败：{e}")

# ===================== GitHub 日志输出 =====================
def print_for_github_email(results):
    print("\n" + "="*60)
    print("📈 股票技术分析报告（GitHub 邮件通知）")
    print("="*60)
    full_text = ""
    for i, s in enumerate(results, 1):
        part = f"""
【{i}】{s['name']} [{s['code']}]
现价：{s['price']}   {s['pct']:+}%
RSI(6)={s['rsi6']}  RSI(12)={s['rsi12']}  RSI(24)={s['rsi24']}
EMA20={s['ema20']}

🤖 AI 专业分析：
{s['ai_analysis']}
{"-"*60}
"""
        print(part)
        full_text += part
    return full_text

# ===================== 主程序 =====================
if __name__ == "__main__":
    bs.login()
    stock_codes = [c.strip() for c in os.getenv("STOCK_LIST", "").split("-") if c.strip()]
    results = []

    for code in stock_codes:
        basic = get_quote(code)
        closes = get_daily(code)
        if len(closes) < 20:
            continue

        stock = {
            "code": code,
            "name": basic["name"],
            "price": basic["price"],
            "pct": basic["pct"],
            "rsi6": rsi(closes, 6),
            "rsi12": rsi(closes, 12),
            "rsi24": rsi(closes, 24),
            "ema20": ema(closes, 20)
        }
        stock["ai_analysis"] = ai_analyze(stock)
        results.append(stock)

    # 输出日志并获取完整内容
    full_content = print_for_github_email(results)

    # 双平台推送
    push_to_pushdeer(full_content)
    push_to_serverchan(full_content)

    bs.logout()
    print("\n🎉 全部任务执行完成")
