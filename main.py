import os
import datetime
import requests
import numpy as np
import pandas as pd
import baostock as bs
from openai import OpenAI

# ==============================================
# 超级详细日志系统（控制台 + 文件）
# ==============================================
log_lines = []
log_file_path = ""

def log(msg, level="DEBUG"):
    global log_lines
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{time_str}] [{level.upper()}] {msg}"
    print(line)
    log_lines.append(line)

def save_log_to_file():
    global log_file_path
    date_str = datetime.date.today().strftime("%Y%m%d")
    filename = f"stock_log_{date_str}.txt"
    log_file_path = os.path.abspath(filename)
    log(f"日志文件将保存到：{log_file_path}")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"✅ 日志已保存：{log_file_path}")
    return filename

# ==============================================
# 打印所有环境变量（超级DEBUG）
# ==============================================
def dump_env():
    log("===== 环境变量全量输出 =====")
    env_list = [
        "STOCK_LIST",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_MODEL",
        "PUSHDEER_TOKEN",
        "SERVERCHAN_KEY"
    ]
    for k in env_list:
        v = os.getenv(k)
        if v:
            log(f"环境变量 {k} = {v[:6]}...（已隐藏后半段）")
        else:
            log(f"环境变量 {k} = 未配置", "WARN")
    log("============================")

# ==============================================
# 指标计算
# ==============================================
def rsi(prices, n):
    log(f"RSI{n} 输入长度 = {len(prices)}")
    if len(prices) < n + 1:
        log(f"RSI{n} 数据不足，返回 50")
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g = np.mean(gains[:n])
    avg_l = np.mean(losses[:n])
    for i in range(n, len(gains)):
        avg_g = (avg_g * (n-1) + gains[i]) / n
        avg_l = (avg_l * (n-1) + losses[i]) / n
    rs = avg_g / avg_l if avg_l != 0 else 999
    res = round(100 - 100/(1+rs), 2)
    log(f"RSI{n} = {res}")
    return res

def ema(prices, n=20):
    log(f"EMA{n} 输入长度 = {len(prices)}")
    if len(prices) < n:
        log(f"EMA{n} 数据不足，返回 0")
        return 0.0
    val = round(pd.Series(prices).ewm(span=n, adjust=False).mean().iloc[-1], 2)
    log(f"EMA{n} = {val}")
    return val

# ==============================================
# 数据获取
# ==============================================
def get_daily(code):
    log(f"开始获取日线：{code}")
    try:
        bs_code = f"{code[:2]}.{code[2:]}"
        end = datetime.datetime.now().strftime("%Y-%m-%d")
        start = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime("%Y-%m-%d")
        log(f"Baostock 代码：{bs_code}，日期范围 {start} ~ {end}")
        rs = bs.query_history_k_data_plus(bs_code, "date,close", start, end, "d", "3")
        log(f"查询返回码：{rs.error_code}，信息：{rs.error_msg}")
        df = rs.get_data()
        log(f"返回数据形状：{df.shape}")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        closes = df["close"].dropna().tolist()[-60:]
        log(f"最终有效收盘价数量：{len(closes)}")
        return closes
    except Exception as e:
        log(f"获取日线失败：{str(e)}", "ERROR")
        return []

def get_quote(code):
    log(f"获取实时行情：{code}")
    try:
        url = f"https://qt.gtimg.cn/q={code}"
        log(f"请求URL：{url}")
        res = requests.get(url, timeout=8)
        log(f"响应状态码：{res.status_code}")
        arr = res.text.split("~")
        name = arr[1] if len(arr) > 1 else code
        price = float(arr[3]) if len(arr) > 3 else 0.0
        pre = float(arr[4]) if len(arr) > 4 else price
        pct = round((price - pre) / pre * 100, 2) if pre != 0 else 0
        log(f"名称：{name}，价格：{price}，涨跌幅：{pct}%")
        return {"name": name, "price": price, "pct": pct}
    except Exception as e:
        log(f"获取行情失败：{str(e)}", "ERROR")
        return {"name": code, "price": 0.0, "pct": 0.0}

# ==============================================
# AI 双引擎：DeepSeek + OpenAI第三方
# ==============================================
def ai_analyze(stock):
    prompt = f"""请以资深股票技术分析师身份，对【{stock['name']}/{stock['code']}】进行技术面分析，重点聚焦价格位置、K线形态、量价关系、支撑压力、均线趋势，简要结合行业与政策环境，语言专业精炼，形成投资建议。
分析框架：
股价当前所处区间（高位/中位/低位），关键支撑位与压力位
K线形态、趋势结构、均线系统表现
成交量配合情况
行业景气度及相关政策简要影响
技术面观点+风险提示

指标：
现价：{stock['price']}
涨跌幅：{stock['pct']}%
RSI6：{stock['rsi6']}
RSI12：{stock['rsi12']}
RSI24：{stock['rsi24']}
EMA20：{stock['ema20']}
"""
    log(f"生成Prompt长度：{len(prompt)}")

    # 先试 DeepSeek
    try:
        log("尝试调用 DeepSeek API")
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise Exception("未配置 DEEPSEEK_API_KEY")
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200
        )
        content = resp.choices[0].message.content.strip()
        log("✅ DeepSeek 调用成功")
        log(f"AI返回长度：{len(content)}")
        return content
    except Exception as e:
        log(f"DeepSeek 调用失败：{str(e)}", "ERROR")

    # 降级第三方 OpenAI
    try:
        log("尝试调用第三方 OpenAI 兼容接口")
        key = os.getenv("OPENAI_API_KEY")
        base = os.getenv("OPENAI_API_BASE")
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        log(f"OPENAI_API_BASE={base}, model={model}")
        if not key or not base:
            raise Exception("第三方 OpenAI 未配置")
        client = OpenAI(api_key=key, base_url=base)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200
        )
        content = resp.choices[0].message.content.strip()
        log("✅ 第三方 OpenAI 调用成功")
        log(f"AI返回长度：{len(content)}")
        return content
    except Exception as e:
        log(f"第三方 OpenAI 失败：{str(e)}", "ERROR")
        return f"AI 分析不可用：{str(e)}"

# ==============================================
# 推送：PushDeer
# ==============================================
def push_to_pushdeer(content):
    log("开始推送 PushDeer")
    token = os.getenv("PUSHDEER_TOKEN")
    if not token:
        log("PushDeer 未配置，跳过", "WARN")
        return
    try:
        url = "https://api2.pushdeer.com/message/push"
        log(f"推送URL：{url}")
        res = requests.get(url, params={"pushkey": token, "text": content}, timeout=10)
        log(f"PushDeer 响应：{res.status_code} {res.text}")
        log("✅ PushDeer 推送成功")
    except Exception as e:
        log(f"PushDeer 推送失败：{str(e)}", "ERROR")

# ==============================================
# 推送：Server酱
# ==============================================
def push_to_serverchan(content):
    log("开始推送 Server酱")
    key = os.getenv("SERVERCHAN_KEY")
    if not key:
        log("Server酱 未配置，跳过", "WARN")
        return
    log(f"Server酱 KEY 长度：{len(key)}")
    try:
        url = f"https://sctapi.ftqq.com/{key}.send"
        data = {"title": "📈 股票技术分析报告", "desp": content}
        log(f"请求URL：{url}")
        res = requests.post(url, data=data, timeout=15)
        log(f"Server酱 响应：{res.status_code} {res.text}")
        log("✅ Server酱 推送成功")
    except Exception as e:
        log(f"Server酱 推送失败：{str(e)}", "ERROR")

# ==============================================
# GitHub 强制邮件输出（必发）
# ==============================================
def build_github_email_content(results):
    log("构建 GitHub 邮件内容")
    content = "📈 股票技术分析报告\n\n"
    for s in results:
        part = f"""
【{s['name']} {s['code']}】
现价：{s['price']} {s['pct']:+}%
RSI6={s['rsi6']} RSI12={s['rsi12']} RSI24={s['rsi24']} EMA20={s['ema20']}
AI分析：{s['ai_analysis']}
----------------------------------------
"""
        content += part
    log("邮件内容构建完成")
    return content

def send_to_github_notice(content):
    log("发送 GitHub Notice 触发邮件")
    print("\n----------------------------------------")
    print("::notice::" + content.replace("\n", " | "))
    print("----------------------------------------\n")
    log("✅ GitHub 邮件通知已触发")

# ==============================================
# 主程序
# ==============================================
if __name__ == "__main__":
    log("========== 程序启动 ==========")
    dump_env()

    bs.login()
    log("Baostock 登录完成")

    raw = os.getenv("STOCK_LIST", "")
    codes = [c.strip() for c in raw.split("-") if c.strip()]
    log(f"股票列表：{codes}")

    results = []
    for code in codes:
        log(f"\n===== 处理 {code} =====")
        basic = get_quote(code)
        closes = get_daily(code)
        if len(closes) < 20:
            log(f"{code} 数据不足，跳过", "ERROR")
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

    full_report = build_github_email_content(results)
    send_to_github_notice(full_report)

    push_to_pushdeer(full_report)
    push_to_serverchan(full_report)

    bs.logout()
    log("Baostock 登出成功")

    save_log_to_file()
    log("========== 全部任务完成 ==========")
