import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np


ts.set_token('7a330a817fc8482d3c9c04cf5cae101965298c9b904ca484c01bfde8')
pro = ts.pro_api()

# ========== 1. 获取数据（分两次获取，然后合并） ==========
stock_code = '600519.SH'  #
start_date = '20250703'
end_date = '20260703'

# 1.1 获取日线行情（开、高、低、收、成交量）
df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
df = df.sort_values('trade_date').reset_index(drop=True)

# 1.2 获取每日估值和换手率（pe, pb, turnover_rate）
df_basic = pro.daily_basic(ts_code=stock_code, start_date=start_date, end_date=end_date,
                           fields='trade_date,pe,pb,turnover_rate')
df_basic = df_basic.sort_values('trade_date').reset_index(drop=True)

# 1.3 合并两个DataFrame（按交易日期）
df = pd.merge(df, df_basic, on='trade_date', how='left')

# 如果某天没有估值数据（比如停牌），用前一天的填充（或直接删除，这里用前向填充）
df.ffill(inplace=True)

# 转换日期格式，设为索引
df['trade_date'] = pd.to_datetime(df['trade_date'])
df.set_index('trade_date', inplace=True)

# 保存CSV（作业要求）
df.to_csv('stock_data.csv', encoding='utf-8-sig')

print(f"数据获取成功！共 {len(df)} 个交易日")

# ========== 2. 计算技术指标 ==========
# 均线
df['MA5'] = df['close'].rolling(5).mean()
df['MA20'] = df['close'].rolling(20).mean()
df['MA60'] = df['close'].rolling(60).mean()

# MACD
def calc_macd(data):
    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    dif = exp1 - exp2
    dea = dif.ewm(span=9, adjust=False).mean()
    macd = (dif - dea) * 2
    return dif, dea, macd
df['DIF'], df['DEA'], df['MACD'] = calc_macd(df)

# RSI
def calc_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
df['RSI'] = calc_rsi(df)

# KDJ（简化版）
low_min = df['low'].rolling(9).min()
high_max = df['high'].rolling(9).max()
df['RSV'] = (df['close'] - low_min) / (high_max - low_min) * 100
df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
df['D'] = df['K'].ewm(com=2, adjust=False).mean()
df['J'] = 3 * df['K'] - 2 * df['D']

# 布林带（以MA20为中轨）
df['BB_upper'] = df['MA20'] + 2 * df['close'].rolling(20).std()
df['BB_lower'] = df['MA20'] - 2 * df['close'].rolling(20).std()

# ========== 3. 开始画图 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 图1：股价走势 + 均线
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['close'], label='收盘价', linewidth=1.5)
plt.plot(df.index, df['MA5'], label='MA5', linestyle='--', linewidth=1)
plt.plot(df.index, df['MA20'], label='MA20', linestyle='--', linewidth=1)
plt.plot(df.index, df['MA60'], label='MA60', linestyle='-.', linewidth=1)
plt.title(f'{stock_code} 股价走势与均线分析')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('图1_股价走势与均线.png', dpi=300, bbox_inches='tight')
plt.close()

# 图2：K线蜡烛图（使用mplfinance）
# 【修复写法】先画出图，返回图表对象 fig
fig, axlist = mpf.plot(df, type='candle', volume=False, style='charles', returnfig=True)

# 单独给图表的标题赋值，并强制指定使用黑体（SimHei）字体
axlist[0].set_title(f'{stock_code} K线蜡烛图', fontproperties='SimHei')

# 保存图片并关闭画布释放内存
fig.savefig('图2_K线蜡烛图.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 图3：MACD
plt.figure(figsize=(12, 4))
plt.plot(df.index, df['DIF'], label='DIF', linewidth=1.5)
plt.plot(df.index, df['DEA'], label='DEA', linewidth=1.5)
plt.bar(df.index, df['MACD'], label='MACD柱', width=0.8, alpha=0.5)
plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
plt.title('MACD指标分析', fontproperties='SimHei')  # 强制加字体
plt.legend(prop={'family': 'SimHei'})               # 图例强制加字体
plt.grid(True, alpha=0.3)
plt.savefig('图3_MACD指标.png', dpi=300, bbox_inches='tight')
plt.close()

# 图4：RSI
plt.figure(figsize=(12, 4))
plt.plot(df.index, df['RSI'], color='purple', linewidth=1.5)
plt.axhline(70, color='red', linestyle='--', linewidth=0.8)
plt.axhline(30, color='green', linestyle='--', linewidth=0.8)
plt.ylim(0, 100)
plt.title('RSI相对强弱指标分析', fontproperties='SimHei') # 强制加字体
plt.grid(True, alpha=0.3)
plt.savefig('图4_RSI指标.png', dpi=300, bbox_inches='tight')
plt.close()

# 图5：KDJ
plt.figure(figsize=(12, 4))
plt.plot(df.index, df['K'], label='K值', linewidth=1.5)
plt.plot(df.index, df['D'], label='D值', linewidth=1.5)
plt.plot(df.index, df['J'], label='J值', linestyle='--', linewidth=1)
plt.axhline(100, color='red', linestyle='--', linewidth=0.5)
plt.axhline(0, color='green', linestyle='--', linewidth=0.5)
plt.title('KDJ随机指标分析', fontproperties='SimHei')  # 强制加字体
plt.legend(prop={'family': 'SimHei'})                 # 图例强制加字体
plt.grid(True, alpha=0.3)
plt.savefig('图5_KDJ指标.png', dpi=300, bbox_inches='tight')
plt.close()

# 图6：布林带
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['close'], label='收盘价', linewidth=1.5)
plt.plot(df.index, df['MA20'], label='中轨(MA20)', linewidth=1)
plt.plot(df.index, df['BB_upper'], label='上轨', linestyle='--', linewidth=1, color='red')
plt.plot(df.index, df['BB_lower'], label='下轨', linestyle='--', linewidth=1, color='green')
plt.fill_between(df.index, df['BB_upper'], df['BB_lower'], alpha=0.1)
plt.title('布林带指标分析', fontproperties='SimHei')  # 强制加字体
plt.legend(prop={'family': 'SimHei'})                # 图例强制加字体
plt.grid(True, alpha=0.3)
plt.savefig('图6_布林带指标.png', dpi=300, bbox_inches='tight')
plt.close()

# 图7：估值指标（PE和PB双轴）
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()
ax1.plot(df.index, df['pe'], label='PE', color='blue', linewidth=1.5)
ax2.plot(df.index, df['pb'], label='PB', color='orange', linewidth=1.5)
ax1.set_xlabel('日期', fontproperties='SimHei')               # 强制加字体
ax1.set_ylabel('PE', color='blue', fontproperties='SimHei') # 强制加字体
ax2.set_ylabel('PB', color='orange', fontproperties='SimHei')# 强制加字体
plt.title('估值指标分析 (PE / PB)', fontproperties='SimHei')   # 强制加字体
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), prop={'family': 'SimHei'}) # 这里使用 prop
plt.grid(True, alpha=0.3)
plt.savefig('图7_估值指标.png', dpi=300, bbox_inches='tight')
plt.close()

# 图8：换手率
plt.figure(figsize=(12, 4))
plt.bar(df.index, df['turnover_rate'], label='换手率', width=0.8, alpha=0.6)
plt.axhline(df['turnover_rate'].mean(), color='red', linestyle='--', label='平均换手率')
plt.title('换手率分析', fontproperties='SimHei')  # 强制加字体
plt.legend(prop={'family': 'SimHei'})             # 图例强制加字体
plt.grid(True, alpha=0.3)
plt.savefig('图8_换手率.png', dpi=300, bbox_inches='tight')
plt.close()

print("所有图表已生成完毕！共8张图片 + 1个CSV文件。")

# 在代码最末尾（画完所有图之后）加上这一段，运行后会直接打印出你需要填的数据
print("\n" + "="*50)
print("【请将以下数据填到解读模板中】")
print("="*50)
latest = df.iloc[-1]
print(f"1. 最新收盘价: {latest['close']:.2f} 元")
print(f"2. 期初价格 (第一行): {df.iloc[0]['close']:.2f} 元")
print(f"3. 期间最高价: {df['close'].max():.2f} 元")
print(f"4. 期间最低价: {df['close'].min():.2f} 元")
print(f"5. 涨跌幅: {(latest['close']/df.iloc[0]['close'] - 1)*100:.2f}%")
print(f"6. MA5: {latest['MA5']:.2f} 元")
print(f"7. MA20: {latest['MA20']:.2f} 元")
print(f"8. MA60: {latest['MA60']:.2f} 元")
print(f"9. RSI(14): {latest['RSI']:.2f}")
print(f"10. MACD柱状值: {latest['MACD']:.3f}")
print(f"11. 布林带上轨: {latest['BB_upper']:.2f} 元")
print(f"12. 布林带下轨: {latest['BB_lower']:.2f} 元")
print(f"13. PE(市盈率): {latest['pe']:.2f} 倍")
print(f"14. PB(市净率): {latest['pb']:.2f} 倍")
print(f"15. 换手率: {latest['turnover_rate']:.2f}%")
print("="*50)