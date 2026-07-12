# hw03.py - 北大光华量化交易 TASK3 完整实现
# 特性：自动数据获取（Tushare/本地/模拟），双均线策略回测，多图输出

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib import rcParams

# ==================== 1. 全局设置 ====================
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# 股票代码
STOCK_CODE = '000518.SZ'   # 四环生物
SHORT_WIN = 5
LONG_WIN = 20

# ==================== 2. 数据获取（含自动降级） ====================
def get_stock_data(code, start='20230101', end='20260711'):
    """获取股票数据，若失败则返回模拟数据"""
    csv_file = 'stock_data.csv'
    
    # 尝试从本地加载
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            if 'close' in df.columns and not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date').set_index('trade_date')
                print(f"✅ 成功加载本地数据：{csv_file}，共 {len(df)} 条记录")
                return df
            else:
                print("⚠️ 本地数据文件损坏，将重新获取")
                os.remove(csv_file)
        except Exception as e:
            print(f"⚠️ 读取本地文件失败：{e}，将重新获取")
            try:
                os.remove(csv_file)
            except:
                pass
    
    # 尝试从 Tushare 下载
    try:
        from config import TUSHARE_TOKEN
        import tushare as ts
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()
        print(f"🔄 正在从 Tushare 下载 {code} 数据...")
        df = pro.daily(ts_code=code, start_date=start, end_date=end)
        if df.empty:
            raise ValueError("Tushare 返回空数据")
        df = df.sort_values('trade_date')
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        df.to_csv(csv_file)
        print(f"✅ Tushare 数据下载成功，共 {len(df)} 条记录，已保存至本地")
        return df
    except Exception as e:
        print(f"⚠️ Tushare 数据获取失败：{e}")
        print("🔄 将使用模拟数据（仅供演示策略逻辑，不代表真实市场表现）")
    
    # 最终降级：生成模拟数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2026-07-11', freq='B')
    prices = 10 + np.cumsum(np.random.randn(len(dates)) * 0.2)
    df = pd.DataFrame({'close': prices}, index=dates)
    df.to_csv(csv_file)
    print(f"✅ 模拟数据已生成，共 {len(df)} 条记录")
    return df

# ==================== 3. 策略回测函数 ====================
def run_dual_ma_strategy(data, short_win, long_win):
    df = data[['close']].copy()
    df[f'MA_{short_win}'] = df['close'].rolling(short_win).mean()
    df[f'MA_{long_win}'] = df['close'].rolling(long_win).mean()
    
    # 信号
    df['signal'] = np.where(df[f'MA_{short_win}'] > df[f'MA_{long_win}'], 1, 0)
    df['position'] = df['signal'].diff()
    
    # 模拟交易
    initial_capital = 100000.0
    commission = 0.0005
    capital = initial_capital
    shares = 0.0
    portfolio = []
    
    for i in range(len(df)):
        price = df['close'].iloc[i]
        pos_signal = df['position'].iloc[i]
        if pos_signal == 1 and capital > 0:
            shares = capital * (1 - commission) / price
            capital = 0.0
        elif pos_signal == -1 and shares > 0:
            capital = shares * price * (1 - commission)
            shares = 0.0
        portfolio.append(capital + shares * price)
    
    df['portfolio_value'] = portfolio
    df['returns'] = df['portfolio_value'].pct_change()
    
    # 指标
    final_value = df['portfolio_value'].iloc[-1]
    cum_ret = (final_value - initial_capital) / initial_capital
    
    df['cummax'] = df['portfolio_value'].cummax()
    df['drawdown'] = (df['cummax'] - df['portfolio_value']) / df['cummax']
    mdd = df['drawdown'].max()
    
    risk_free = 0.03
    excess_ret = df['returns'] - risk_free / 252
    sharpe = np.sqrt(252) * excess_ret.mean() / excess_ret.std() if excess_ret.std() != 0 else 0
    
    trades = df['position'].abs().sum()
    
    print(f"\n📊 参数 {short_win}/{long_win} 回测结果：")
    print(f"   累计回报: {cum_ret:.2%}")
    print(f"   最大回撤: {mdd:.2%}")
    print(f"   夏普比率: {sharpe:.4f}")
    print(f"   交易信号次数: {int(trades)}")
    
    return df, cum_ret, mdd, sharpe

# ==================== 4. 绘图函数 ====================
def plot_results(data1, data2, params1=(5,20), params2=(10,30), stock_name='股票'):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{stock_name} 双均线策略回测报告', fontsize=18, fontweight='bold')
    
    # 子图1：短周期信号
    ax1 = axes[0, 0]
    ax1.plot(data1.index, data1['close'], label='收盘价', color='black', alpha=0.6, linewidth=1)
    ax1.plot(data1.index, data1[f'MA_{params1[0]}'], label=f'{params1[0]}日均线', color='blue')
    ax1.plot(data1.index, data1[f'MA_{params1[1]}'], label=f'{params1[1]}日均线', color='red')
    buy1 = data1[data1['position'] == 1]
    sell1 = data1[data1['position'] == -1]
    ax1.scatter(buy1.index, buy1['close'], marker='^', color='green', s=80, label='买入(金叉)')
    ax1.scatter(sell1.index, sell1['close'], marker='v', color='red', s=80, label='卖出(死叉)')
    ax1.set_title(f'图1：{params1[0]}/{params1[1]} 均线交叉信号')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 子图2：短周期净值
    ax2 = axes[0, 1]
    ax2.plot(data1.index, data1['portfolio_value'], label='策略净值', color='darkgreen', linewidth=2)
    ax2.axhline(y=100000, color='gray', linestyle='--', label='初始本金')
    # 获取指标（由于函数返回，我们通过全局变量或重新计算，这里简单起见，在外部传入）
    # 但我们在调用时动态计算，更干净：在外部把指标作为参数传入，但为简化，我们直接根据data1计算一次
    # 使用预计算的指标
    ax2.set_title(f'图2：{params1[0]}/{params1[1]} 净值曲线')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 子图3：长周期信号
    ax3 = axes[1, 0]
    ax3.plot(data2.index, data2['close'], label='收盘价', color='black', alpha=0.6, linewidth=1)
    ax3.plot(data2.index, data2[f'MA_{params2[0]}'], label=f'{params2[0]}日均线', color='blue')
    ax3.plot(data2.index, data2[f'MA_{params2[1]}'], label=f'{params2[1]}日均线', color='red')
    buy2 = data2[data2['position'] == 1]
    sell2 = data2[data2['position'] == -1]
    ax3.scatter(buy2.index, buy2['close'], marker='^', color='green', s=80, label='买入(金叉)')
    ax3.scatter(sell2.index, sell2['close'], marker='v', color='red', s=80, label='卖出(死叉)')
    ax3.set_title(f'图3：{params2[0]}/{params2[1]} 均线交叉信号')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 子图4：长周期净值
    ax4 = axes[1, 1]
    ax4.plot(data2.index, data2['portfolio_value'], label='策略净值', color='darkblue', linewidth=2)
    ax4.axhline(y=100000, color='gray', linestyle='--', label='初始本金')
    ax4.set_title(f'图4：{params2[0]}/{params2[1]} 净值曲线')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('双均线策略回测报告.png', dpi=300)
    print("\n✅ 图表已保存为：双均线策略回测报告.png")
    plt.show()

# ==================== 5. 主程序 ====================
if __name__ == "__main__":
    print("=" * 50)
    print("北大光华量化交易 TASK3 策略回测")
    print("=" * 50)
    
    # 获取数据
    df = get_stock_data(STOCK_CODE)
    print(f"数据区间：{df.index.min()} 至 {df.index.max()}，共 {len(df)} 个交易日")
    
    # 运行两组参数
    data1, ret1, mdd1, sharpe1 = run_dual_ma_strategy(df, 5, 20)
    data2, ret2, mdd2, sharpe2 = run_dual_ma_strategy(df, 10, 30)
    
    # 将指标添加到dataframe中以便绘图时展示（可选）
    # 我们直接绘图，标题中不动态显示指标，但可在图中加文本，简单起见保持现有
    # 但为了显示指标，我们可以修改子图标题
    # 重新设置子图标题（通过修改axes）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{STOCK_CODE} 双均线策略回测报告', fontsize=18, fontweight='bold')
    
    ax1 = axes[0, 0]
    ax1.plot(data1.index, data1['close'], label='收盘价', color='black', alpha=0.6, linewidth=1)
    ax1.plot(data1.index, data1['MA_5'], label='5日均线', color='blue')
    ax1.plot(data1.index, data1['MA_20'], label='20日均线', color='red')
    buy1 = data1[data1['position'] == 1]
    sell1 = data1[data1['position'] == -1]
    ax1.scatter(buy1.index, buy1['close'], marker='^', color='green', s=80, label='买入(金叉)')
    ax1.scatter(sell1.index, sell1['close'], marker='v', color='red', s=80, label='卖出(死叉)')
    ax1.set_title(f'图1：5/20 均线交叉信号 (累计收益:{ret1:.2%})')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(data1.index, data1['portfolio_value'], label='策略净值', color='darkgreen', linewidth=2)
    ax2.axhline(y=100000, color='gray', linestyle='--', label='初始本金')
    ax2.set_title(f'图2：5/20 净值曲线 (MDD:{mdd1:.2%} 夏普:{sharpe1:.2f})')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.plot(data2.index, data2['close'], label='收盘价', color='black', alpha=0.6, linewidth=1)
    ax3.plot(data2.index, data2['MA_10'], label='10日均线', color='blue')
    ax3.plot(data2.index, data2['MA_30'], label='30日均线', color='red')
    buy2 = data2[data2['position'] == 1]
    sell2 = data2[data2['position'] == -1]
    ax3.scatter(buy2.index, buy2['close'], marker='^', color='green', s=80, label='买入(金叉)')
    ax3.scatter(sell2.index, sell2['close'], marker='v', color='red', s=80, label='卖出(死叉)')
    ax3.set_title(f'图3：10/30 均线交叉信号 (累计收益:{ret2:.2%})')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.plot(data2.index, data2['portfolio_value'], label='策略净值', color='darkblue', linewidth=2)
    ax4.axhline(y=100000, color='gray', linestyle='--', label='初始本金')
    ax4.set_title(f'图4：10/30 净值曲线 (MDD:{mdd2:.2%} 夏普:{sharpe2:.2f})')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('四环生物_双均线策略回测报告.png', dpi=300)
    print("\n✅ 最终图表已保存为：四环生物_双均线策略回测报告.png")
    plt.show()