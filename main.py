import os
import requests
import datetime
from openai import OpenAI

# ===================== 从 GitHub Secrets 自动读取 =====================
# 读取 3 个密钥，代码里完全不暴露信息
FEISHU_WEBHOOK_URL = os.getenv("FEISHU_WEBHOOK_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
STOCK_LIST = os.getenv("STOCK_LIST").split("-")  # 自动按 - 分割股票
# =====================================================================

def is_trade_day():
    today = datetime.date.today().strftime("%Y%m%d")
    try:
        resp = requests.get("https://api.apihubs.cn/holiday/get", timeout=10)
        data = resp.json()
        for d in data["data"]["list"]:
            if d["date"] == today:
                return d["workday"] == 1
        return False
    except:
        return False

def get_stock(code):
    url = f"https://hq.sinajs.cn/list={code.lower()}"
    res = requests.get(url, timeout=5)
    arr = res.text.split('"')[1].split(",")
    name = arr[0]
    pre = float(arr[2])
    now = float(arr[3])
    high = float(arr[4])
    low = float(arr[5])
    pct = round((now - pre) / pre * 100, 2)
    return {
        "name": name, "code": code, "now": now,
        "pct": pct, "high": high, "low": low
    }

def ai_analysis(stocks):
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    prompt = f"""
今天是{datetime.date.today()}，请对以下股票做简短专业分析：
1.今日表现
2.短期走势
3.简要建议

数据：
"""
    for s in stocks:
        prompt += f"【{s['name']}】现价{s['now']}，涨跌幅{s['pct']}%\n"
    
    prompt += "\n语言简洁、分点、不啰嗦。"
    
    res = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return res.choices[0].message.content

def send_feishu(msg):
    requests.post(FEISHU_WEBHOOK_URL, json={
        "msg_type": "text",
        "content": {"text": msg}
    }, timeout=10)

if __name__ == "__main__":
    if not is_trade_day():
        print("非交易日")
        exit()

    # 获取所有股票数据
    stocks = []
    for code in STOCK_LIST:
        stocks.append(get_stock(code))

    # 组装消息
    msg = f"📈 A股实时提醒 {datetime.date.today()}\n\n"
    for s in stocks:
        msg += f"【{s['name']}】{s['code']}\n现价：{s['now']}  涨跌：{s['pct']}%\n\n"
    
    # AI 分析
    msg += "🤖 DeepSeek AI 分析：\n"
    msg += ai_analysis(stocks)

    # 推送飞书
    send_feishu(msg)
    print("推送成功")
