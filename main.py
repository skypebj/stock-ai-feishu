import os
import datetime
import requests
import numpy as np
import pandas as pd
import baostock as bs
from openai import OpenAI

# ==============================================
# 日志系统（控制台 + 文件保存）
# ==============================================
log_lines = []
def log(msg, level="INFO"):
    time_str = datetime.datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO":"[INFO]","ERROR":"[ERROR]","DEBUG":"[DEBUG]"}.get(level,"[INFO]")
    line = f"{time_str} {prefix} {msg}"
    print(line)
    log_lines.append(line)

def save_log():
    date_str = datetime.date.today().strftime("%Y%m%d")
    filename = f"stock_log_{date_str}.txt"
    with open(filename,"w",encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"日志已保存：{filename}")
    return filename

# ==============================================
# 技术指标
# ==============================================
def rsi(prices,n):
    log(f"RSI{n} 数据量：{len(prices)}","DEBUG")
    if len(prices)<n+1:return 50.0
    deltas=np.diff(prices)
    gains=np.where(deltas>0,deltas,0.0)
    losses=np.where(deltas<0,-deltas,0.0)
    avg_g=np.mean(gains[:n])
    avg_l=np.mean(losses[:n])
    for i in range(n,len(gains)):
        avg_g=(avg_g*(n-1)+gains[i])/n
        avg_l=(avg_l*(n-1)+losses[i])/n
    rs=avg_g/avg_l if avg_l!=0 else 999
    return round(100-100/(1+rs),2)

def ema(prices,n=20):
    log(f"EMA{n} 数据量：{len(prices)}","DEBUG")
    if len(prices)<n:return 0.0
    return round(pd.Series(prices).ewm(span=n,adjust=False).mean().iloc[-1],2)

# ==============================================
# 数据获取
# ==============================================
def get_daily(code):
    log(f"获取日线：{code}")
    try:
        bs_code=f"{code[:2]}.{code[2:]}"
        end=datetime.datetime.now().strftime("%Y-%m-%d")
        start=(datetime.datetime.now()-datetime.timedelta(days=120)).strftime("%Y-%m-%d")
        rs=bs.query_history_k_data_plus(bs_code,"date,close",start,end,"d","3")
        df=rs.get_data()
        df["close"]=pd.to_numeric(df["close"],errors="coerce")
        closes=df["close"].dropna().tolist()[-60:]
        log(f"有效K线：{len(closes)}","DEBUG")
        return closes
    except Exception as e:
        log(f"日线失败：{e}","ERROR")
        return []

def get_quote(code):
    log(f"获取行情：{code}")
    try:
        txt=requests.get(f"https://qt.gtimg.cn/q={code}",timeout=5).text
        arr=txt.split("~")
        name=arr[1]
        p=float(arr[3])
        pre=float(arr[4])
        return {"name":name,"price":p,"pct":round((p-pre)/pre*100,2)}
    except Exception as e:
        log(f"行情失败：{e}","ERROR")
        return {"name":code,"price":0,"pct":0}

# ==============================================
# AI 双引擎自动切换（DeepSeek → 第三方OpenAI兼容）
# ==============================================
def ai_analyze(stock):
    prompt=f"""请以资深股票技术分析师身份，对【{stock['name']}/{stock['code']}】进行技术面分析，重点聚焦价格位置、K线形态、量价关系、支撑压力、均线趋势，简要结合行业与政策环境，语言专业精炼，形成投资建议。
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
    # 优先 DeepSeek
    try:
        log("尝试调用 DeepSeek")
        k=os.getenv("DEEPSEEK_API_KEY")
        if not k:raise Exception("无DeepSeek Key")
        c=OpenAI(api_key=k,base_url="https://api.deepseek.com")
        r=c.chat.completions.create(model="deepseek-chat",messages=[{"role":"user","content":prompt}],temperature=0.3,max_tokens=1000)
        log("✅ DeepSeek 调用成功")
        return r.choices[0].message.content.strip()
    except Exception as e:
        log(f"DeepSeek 失败：{e}，切换第三方OpenAI","ERROR")
    # 降级 OpenAI 第三方
    try:
        log("尝试调用第三方OpenAI兼容接口")
        k=os.getenv("OPENAI_API_KEY")
        b=os.getenv("OPENAI_API_BASE")
        m=os.getenv("OPENAI_MODEL","gpt-3.5-turbo")
        if not k or not b:raise Exception("第三方OpenAI未配置")
        c=OpenAI(api_key=k,base_url=b)
        r=c.chat.completions.create(model=m,messages=[{"role":"user","content":prompt}],temperature=0.3,max_tokens=1000)
        log("✅ 第三方OpenAI 调用成功")
        return r.choices[0].message.content.strip()
    except Exception as e:
        log(f"AI全部失败：{e}","ERROR")
        return f"AI分析不可用：{str(e)}"

# ==============================================
# 推送：PushDeer
# ==============================================
def push_to_pushdeer(content):
    t=os.getenv("PUSHDEER_TOKEN")
    log(f"PushDeer 已配置：{t is not None}","DEBUG")
    if not t:
        log("ℹ PushDeer 未配置")
        return
    try:
        requests.get("https://api2.pushdeer.com/message/push",params={"pushkey":t,"text":content},timeout=10)
        log("✅ PushDeer 推送成功")
    except Exception as e:
        log(f"PushDeer 失败：{e}","ERROR")

# ==============================================
# 推送：Server酱
# ==============================================
def push_to_serverchan(content):
    k=os.getenv("SERVERCHAN_KEY")
    log(f"Server酱 已配置：{k is not None}","DEBUG")
    if k:
        log(f"Server酱 Key长度：{len(k)}","DEBUG")
    if not k:
        log("ℹ Server酱 未配置")
        return
    try:
        r=requests.post(f"https://sctapi.ftqq.com/{k}.send",data={"title":"📈股票分析报告","desp":content},timeout=15)
        log(f"Server酱 响应：{r.status_code} {r.text}","DEBUG")
        log("✅ Server酱 推送成功")
    except Exception as e:
        log(f"Server酱 失败：{e}","ERROR")

# ==============================================
# GitHub 邮件通知输出
# ==============================================
def build_and_output_report(results):
    log("输出GitHub通知日志")
    full=""
    for s in results:
        item=f"""
【{s['name']} {s['code']}】
现价：{s['price']} {s['pct']:+}%
RSI6={s['rsi6']} RSI12={s['rsi12']} RSI24={s['rsi24']} EMA20={s['ema20']}
🤖AI分析：{s['ai_analysis']}
------------------------------------------------------------
"""
        print(f"::notice::{item[:200]}...")
        full+=item
    return full

# ==============================================
# 主程序
# ==============================================
if __name__=="__main__":
    log("========== 程序启动 ==========")
    bs.login()
    codes=[c.strip() for c in os.getenv("STOCK_LIST","").split("-") if c.strip()]
    results=[]
    for code in codes:
        basic=get_quote(code)
        closes=get_daily(code)
        if len(closes)<20:
            log(f"{code} 数据不足，跳过","ERROR")
            continue
        stock={
            "code":code,"name":basic["name"],"price":basic["price"],"pct":basic["pct"],
            "rsi6":rsi(closes,6),"rsi12":rsi(closes,12),"rsi24":rsi(closes,24),"ema20":ema(closes,20)
        }
        stock["ai_analysis"]=ai_analyze(stock)
        results.append(stock)
    full=build_and_output_report(results)
    push_to_pushdeer(full)
    push_to_serverchan(full)
    bs.logout()
    log("登出成功")
    save_log()
    log("🎉 全部完成")
