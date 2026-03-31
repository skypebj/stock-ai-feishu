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
    log(f"股票列表原始字符串: {STOCK_LIST_STR}")
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

# ===================== 交易日判断 =====================
def is_trade_day():
    log("开始判断是否为交易日")
    try:
        today = datetime.date.today().strftime("%Y%m%d")
        log(f"今日日期: {today}")
        
        resp = requests.get("https://api.apihubs.cn/holiday/get", timeout=10)
        log(f"交易日历接口状态码: {resp.status_code}")
        
        data = resp.json()
        for d in data.get("data", {}).get("list", []):
            if d.get("date") == today:
                res = d.get("workday") == 1
                log(f"交易日结果: {res}")
                return res
        log("未找到今日日期记录，默认非交易日")
        return False
    except Exception as e:
        error(f"交易日判断失败: {e}")
        return False

# ===================== 获取股票价格 =====================
def get_stock(code):
    log(f"开始获取股票: {code}")
    try:
        url = f"https://hq.sinajs.cn/list={code.lower()}"
        res = requests.get(url, timeout=5)
        log(f"{code} 接口状态码: {res.status_code}")
        
        arr = res.text.split('"')[1].split(",")
        name = arr[0]
        pre = float(arr[2])
        now = float(arr[3])
        pct = round((now - pre) / pre * 100, 2)
        
        success(f"{code} 获取成功: {name} {now}元 {pct}%")
        return {
            "name": name, "code": code,
            "now": now, "pct": pct
        }
    except Exception as e:
        error(f"{code} 获取失败: {e}")
        return {
            "name": "获取失败", "code": code,
            "now": 0, "pct": 0
        }

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
        resp = requests.get(url, timeout=10)
        log(f"PushDeer 响应: {resp.status_code} {resp.text}")
        success("推送完成")
    except Exception as e:
        error(f"推送失败: {e}")

# ===================== 主流程 =====================
if __name__ == "__main__":
    log("进入主流程")
    
    if not is_trade_day():
        log("今日非交易日，程序退出")
        sys.exit(0)
    
    stocks = [get_stock(code) for code in STOCK_LIST]
    log(f"成功获取 {len(stocks)} 只股票")
    
    msg = f"📈 A股实时推送 {datetime.date.today()}\n\n"
    for s in stocks:
        msg += f"【{s['name']}】{s['code']}\n现价：{s['now']}  涨跌：{s['pct']}%\n\n"
    
    msg += "🤖 DeepSeek 分析：\n"
    msg += ai_analysis(stocks)
    
    push_pushdeer(msg)
    
    log("========== 程序全部执行成功 ==========")
