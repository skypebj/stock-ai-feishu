import os
import sys
import requests
import datetime
from openai import OpenAI

# ===================== 日志输出函数 =====================
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

# ===================== 稳定获取股票价格（腾讯接口，不超时） =====================
def get_stock(code):
    log(f"开始获取股票: {code}")
    try:
        code_clean = code.replace(".SH", "").replace(".SZ", "")
        market = "1" if code.endswith(".SH") else "0"
        url = f"https://web.ifzq.gtimg.cn/realstock/quote/{market}{code_clean}.js"
        
        resp = requests.get(url, timeout=15)
        txt = resp.text
        data = txt.split('"')[1].split("~")

        name = data[1]
        now = float(data[3])
        close = float(data[4])
        pct = round((now - close) / close * 100, 2)

        success(f"{code} 获取成功: {name} | {now}元 | {pct}%")
        return {
            "name": name, "code": code,
            "now": now, "pct": pct
        }

    except Exception as e:
        error(f"{code} 获取失败: {str(e)}")
        return {"name": "获取失败", "code": code, "now": 0, "pct": 0}

# ===================== AI 分析 =====================
def ai_analysis(stocks):
    log("开始调用 DeepSeek AI 分析")
    if not DEEPSEEK_API_KEY:
        error("DeepSeek API Key 为空")
        return "未配置 DeepSeek Key"
    
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        prompt = f"今天{datetime.date.today()}，分析以下A股，3点：今日表现、短期走势、操作建议。简洁分点。\n"
        for s in stocks:
            prompt += f"【{s['name']}】{s['code']} 现价{s['now']} 涨跌幅{s['pct']}%\n"
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        success("AI 分析完成")
        return response.choices[0].message.content
    except Exception as e:
        error(f"AI 分析失败: {e}")
        return f"AI 分析异常: {str(e)}"

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
    log("进入主流程（已关闭交易日判断，每日直接执行）")
    
    stocks = [get_stock(code) for code in STOCK_LIST]
    log(f"成功获取 {len(stocks)} 只股票")
    
    msg = f"📈 A股实时推送 {datetime.date.today()}\n\n"
    for s in stocks:
        msg += f"【{s['name']}】{s['code']}\n现价：{s['now']}  涨跌：{s['pct']}%\n\n"
    
    msg += "🤖 DeepSeek 分析：\n"
    msg += ai_analysis(stocks)
    
    push_pushdeer(msg)
    
    log("========== 程序全部执行成功 ==========")
