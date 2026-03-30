import os
import requests
import datetime
from openai import OpenAI

# ===================== 密钥读取 =====================
PUSHDEER_TOKEN = os.getenv("PUSHDEER_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
STOCK_LIST = os.getenv("STOCK_LIST", "").split("-")
# ====================================================

def is_trade_day():
    try:
        today = datetime.date.today().strftime("%Y%m%d")
        resp = requests.get("https://api.apihubs.cn/holiday/get", timeout=10)
        data = resp.json()
        for d in data.get("data", {}).get("list", []):
            if d.get("date") == today:
                return d.get("workday") == 1
    except Exception:
        return False
    return False

def get_stock(code):
    try:
        url = f"https://hq.sinajs.cn/list={code.lower()}"
        res = requests.get(url, timeout=5)
        arr = res.text.split('"')[1].split(",")
        name = arr[0]
        pre = float(arr[2])
        now = float(arr[3])
        pct = round((now - pre) / pre * 100, 2)
        return {"name": name, "code": code, "now": now, "pct": pct}
    except Exception:
        return {"name": "获取失败", "code": code, "now": 0, "pct": 0}

def ai_analysis(stocks):
    if not DEEPSEEK_API_KEY:
        return "DeepSeek API Key 未配置"
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        prompt = f"今天{datetime.date.today()}，分析以下股票，简要3点：今日表现、短期走势、操作建议。\n"
        for s in stocks:
            prompt += f"【{s['name']}】现价{s['now']}，涨跌幅{s['pct']}%\n"
        prompt += "简短、专业、分点。"
        
        res = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"AI分析失败：{str(e)}"

def push_pushdeer(text):
    try:
        text_encoded = requests.utils.quote(text)
        url = f"https://api2.pushdeer.com/message/push?pushkey={PUSHDEER_TOKEN}&text={text_encoded}"
        requests.get(url, timeout=10)
    except Exception:
        pass

if __name__ == "__main__":
    if not is_trade_day():
        print("非交易日")
        exit(0)

    stocks = [get_stock(code) for code in STOCK_LIST if code.strip()]
    
    msg = f"📈 A股实时推送 {datetime.date.today()}\n\n"
    for s in stocks:
        msg += f"【{s['name']}】{s['code']}\n现价：{s['now']}  涨跌：{s['pct']}%\n\n"
    
    msg += "🤖 DeepSeek 分析：\n"
    msg += ai_analysis(stocks)
    
    push_pushdeer(msg)
    print("推送成功")
