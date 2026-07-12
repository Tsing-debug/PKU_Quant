# hw04.py - 北大光华量化交易 TASK4 海龟策略完整实现（四环生物）
# 功能：自动获取数据（本地/Tushare/模拟）、海龟策略回测、多维度可视化

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams

# ==================== 1. 全局设置 ====================
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

STOCK_CODE = '000518.SZ'      # 四环生物
INITIAL_CAPITAL = 100000.0
COMMISSION = 0.0005           # 双边万5

# ==================== 2. 数据获取（含自动降级） ====================
def get_stock_data(code, start='20230101', end='20260711'):
    """
    获取股票日线数据（含high/low/close）
    优先级：本地CSV > Tushare > 模拟数据
    """
    csv_file = 'stock_data.csv'
    
    # 尝试从本地加载
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            required = ['close', 'high', 'low']
            if all(col in df.columns for col in required) and not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date').set_index('trade_date')
                print(f"✅ 本地数据加载成功：{len(df)} 条记录")
                return df
            else:
                print("⚠️ 本地数据缺少必要列，将重新获取")
                os.remove(csv_file)
        except Exception as e:
            print(f"⚠️ 本地数据读取失败：{e}，重新获取")
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
        print(f"🔄 正在从 Tushare 下载 {code} ...")
        df = pro.daily(ts_code=code, start_date=start, end_date=end)
        if df.empty:
            raise ValueError("Tushare 返回空数据")
        df = df.sort_values('trade_date')
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        # 确保有 high/low 列（Tushare 默认有）
        df.to_csv(csv_file)
        print(f"✅ Tushare 下载成功：{len(df)} 条记录")
        return df
    except Exception as e:
        print(f"⚠️ Tushare 失败：{e}")
        print("🔄 使用模拟数据（含 high/low 构造）")
    
    # 模拟数据（构造高波动场景，模拟四环生物）
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2026-07-11', freq='B')
    # 生成价格路径（随机游走，有一定趋势）
    returns = np.random.randn(len(dates)) * 0.02
    prices = 3.0 + np.cumsum(returns)
    prices = np.maximum(prices, 1.5)  # 防止负数
    
    # 构造 high/low
    high = prices * (1 + np.abs(np.random.randn(len(dates)) * 0.015))
    low = prices * (1 - np.abs(np.random.randn(len(dates)) * 0.015))
    
    df = pd.DataFrame({
        'close': prices,
        'high': high,
        'low': low
    }, index=dates)
    
    df.to_csv(csv_file)
    print(f"✅ 模拟数据生成：{len(df)} 条记录")
    return df

# ==================== 3. 海龟策略核心函数 ====================
def run_turtle_strategy(data, entry_period=20, exit_period=10, atr_period=20, stop_mult=2.0):
    """
    海龟策略（多头版本）
    - 入场：收盘价突破 entry_period 日高点（Donchian通道上轨）
    - 出场：收盘价跌破 exit_period 日低点（下轨）或 价格从最高点回撤超过 stop_mult * ATR
    - 返回包含信号、通道、ATR、净值等完整DataFrame
    """
    df = data[['close', 'high', 'low']].copy()
    
    # 1. 高低点通道
    df['high_channel'] = df['high'].rolling(entry_period).max()
    df['low_channel'] = df['low'].rolling(exit_period).min()
    
    # 2. ATR（平均真实波幅）
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(atr_period).mean()
    
    # 3. 信号生成
    df['signal'] = 0          # 0=空仓, 1=买入信号, -1=卖出信号
    df['entry_price'] = np.nan
    df['stop_price'] = np.nan
    
    position = 0
    entry_price = 0.0
    highest_since_entry = 0.0   # 入场后的最高价（用于移动止损）
    
    # 从足够长的数据开始
    start_idx = max(entry_period, exit_period, atr_period)
    for i in range(start_idx, len(df)):
        price = df['close'].iloc[i]
        high_ch = df['high_channel'].iloc[i]
        low_ch = df['low_channel'].iloc[i]
        atr_val = df['atr'].iloc[i]
        
        if position == 0:
            # 空仓，检查是否突破入场
            if price > high_ch:
                position = 1
                entry_price = price
                highest_since_entry = price
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i], 'entry_price'] = entry_price
                df.loc[df.index[i], 'stop_price'] = entry_price - stop_mult * atr_val
        else:
            # 持仓中，更新最高价
            if price > highest_since_entry:
                highest_since_entry = price
            # 计算动态止损价（跟踪止损：最高价回撤 stop_mult * ATR）
            stop_price = highest_since_entry - stop_mult * atr_val
            df.loc[df.index[i], 'stop_price'] = stop_price
            
            # 检查是否触发卖出：跌破低点通道 或 价格跌破止损
            if price < low_ch or price < stop_price:
                position = 0
                df.loc[df.index[i], 'signal'] = -1
                entry_price = 0.0
                highest_since_entry = 0.0
    
    # 生成持仓状态（持仓期间 signal 记为 0，但用 position 列记录）
    df['position'] = df['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    return df

# ==================== 4. 回测与指标计算 ====================
def backtest(df):
    """根据 signal 列进行模拟交易，计算净值及指标"""
    capital = INITIAL_CAPITAL
    shares = 0.0
    portfolio = []
    
    for i in range(len(df)):
        price = df['close'].iloc[i]
        sig = df['signal'].iloc[i]
        
        if sig == 1 and capital > 0:
            # 买入
            shares = capital * (1 - COMMISSION) / price
            capital = 0.0
        elif sig == -1 and shares > 0:
            # 卖出
            capital = shares * price * (1 - COMMISSION)
            shares = 0.0
        
        portfolio.append(capital + shares * price)
    
    df['portfolio_value'] = portfolio
    df['returns'] = df['portfolio_value'].pct_change()
    
    # 指标
    final_val = df['portfolio_value'].iloc[-1]
    cum_ret = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    df['cummax'] = df['portfolio_value'].cummax()
    df['drawdown'] = (df['cummax'] - df['portfolio_value']) / df['cummax']
    mdd = df['drawdown'].max()
    
    risk_free = 0.03
    excess = df['returns'] - risk_free / 252
    if excess.std() != 0:
        sharpe = np.sqrt(252) * excess.mean() / excess.std()
    else:
        sharpe = 0.0
    
    trades = df['signal'].abs().sum()
    
    print(f"   累计回报: {cum_ret:.2%}")
    print(f"   最大回撤: {mdd:.2%}")
    print(f"   夏普比率: {sharpe:.4f}")
    print(f"   交易次数: {int(trades)}")
    
    return df, cum_ret, mdd, sharpe

# ==================== 5. 高级可视化（6合1大图） ====================
def plot_full_report(df1, params1, df2, params2, stock_name="四环生物"):
    """
    绘制包含：信号图、净值曲线、ATR、回撤分布的复合图表
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'{stock_name} 海龟策略完整回测报告', fontsize=20, fontweight='bold')
    
    # 使用 GridSpec 灵活布局
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    # ----- 行1：两组参数信号图 -----
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df1.index, df1['close'], label='收盘价', color='black', alpha=0.6, linewidth=1)
    ax1.plot(df1.index, df1['high_channel'], label=f'{params1[0]}日高点', color='blue', linestyle='--', linewidth=1.2)
    ax1.plot(df1.index, df1['low_channel'], label=f'{params1[1]}日低点', color='red', linestyle='--', linewidth=1.2)
    buy1 = df1[df1['signal'] == 1]
    sell1 = df1[df1['signal'] == -1]
    ax1.scatter(buy1.index, buy1['close'], marker='^', color='green', s=90, label='买入(突破)', zorder=5)
    ax1.scatter(sell1.index, sell1['close'], marker='v', color='red', s=90, label='卖出(破位/止损)', zorder=5)
    ax1.set_title(f'参数组1：入场{params1[0]}日 / 出场{params1[1]}日', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df2.index, df2['close'], label='收盘价', color='black', alpha=0.6, linewidth=1)
    ax2.plot(df2.index, df2['high_channel'], label=f'{params2[0]}日高点', color='blue', linestyle='--', linewidth=1.2)
    ax2.plot(df2.index, df2['low_channel'], label=f'{params2[1]}日低点', color='red', linestyle='--', linewidth=1.2)
    buy2 = df2[df2['signal'] == 1]
    sell2 = df2[df2['signal'] == -1]
    ax2.scatter(buy2.index, buy2['close'], marker='^', color='green', s=90, label='买入', zorder=5)
    ax2.scatter(sell2.index, sell2['close'], marker='v', color='red', s=90, label='卖出', zorder=5)
    ax2.set_title(f'参数组2：入场{params2[0]}日 / 出场{params2[1]}日', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)
    
    # ----- 行2：净值曲线 + ATR -----
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df1.index, df1['portfolio_value'], label='策略净值（参数1）', color='darkgreen', linewidth=2)
    ax3.plot(df2.index, df2['portfolio_value'], label='策略净值（参数2）', color='darkblue', linewidth=2)
    ax3.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5, label='初始本金')
    ax3.set_title('策略净值曲线对比', fontsize=12)
    ax3.legend(loc='best')
    ax3.grid(alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df1.index, df1['atr'], label='ATR (参数1)', color='orange', linewidth=1.5)
    ax4.plot(df2.index, df2['atr'], label='ATR (参数2)', color='purple', linewidth=1.5)
    ax4.set_title('ATR（平均真实波幅）走势', fontsize=12)
    ax4.legend(loc='best')
    ax4.grid(alpha=0.3)
    
    # ----- 行3：回撤分布 + 累计收益对比 -----
    ax5 = fig.add_subplot(gs[2, 0])
    # 绘制回撤曲线（仅显示非零回撤）
    dd1 = df1['drawdown'] * 100
    dd2 = df2['drawdown'] * 100
    ax5.fill_between(df1.index, 0, dd1, color='red', alpha=0.3, label=f'参数1 最大回撤 {dd1.max():.1f}%')
    ax5.fill_between(df2.index, 0, dd2, color='blue', alpha=0.2, label=f'参数2 最大回撤 {dd2.max():.1f}%')
    ax5.set_title('回撤（Drawdown）对比', fontsize=12)
    ax5.set_ylabel('回撤 (%)')
    ax5.legend(loc='best')
    ax5.grid(alpha=0.3)
    
    ax6 = fig.add_subplot(gs[2, 1])
    # 累计收益柱状图（期末值）
    rets = [df1['portfolio_value'].iloc[-1] / INITIAL_CAPITAL - 1, 
            df2['portfolio_value'].iloc[-1] / INITIAL_CAPITAL - 1]
    labels = [f'{params1[0]}/{params1[1]}', f'{params2[0]}/{params2[1]}']
    colors = ['green' if r > 0 else 'red' for r in rets]
    bars = ax6.bar(labels, rets, color=colors, alpha=0.7)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax6.set_title('累计回报对比', fontsize=12)
    ax6.set_ylabel('累计回报')
    for bar, val in zip(bars, rets):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{val:.2%}', ha='center', va='bottom', fontsize=10)
    ax6.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('四环生物_海龟策略完整报告.png', dpi=300, bbox_inches='tight')
    print("\n✅ 完整报告图已保存：四环生物_海龟策略完整报告.png")
    plt.show()

# ==================== 6. 主程序 ====================
if __name__ == "__main__":
    print("=" * 60)
    print(" 北大光华量化交易 TASK4 海龟策略回测系统")
    print("=" * 60)
    
    # 获取数据
    df_raw = get_stock_data(STOCK_CODE)
    print(f"数据时间范围：{df_raw.index.min().date()} 至 {df_raw.index.max().date()}")
    
    # 定义两组参数
    params = [
        (20, 10, 20, 2.0),   # (入场周期, 出场周期, ATR周期, 止损倍数)
        (55, 20, 20, 2.0)    # 长周期版本
    ]
    
    results = []
    dfs = []
    for entry, exit_, atr, mult in params:
        print(f"\n📊 运行参数：入场{entry}日 / 出场{exit_}日 / ATR{atr}日 / 止损{mult}倍ATR")
        df = run_turtle_strategy(df_raw, entry_period=entry, exit_period=exit_, 
                                  atr_period=atr, stop_mult=mult)
        df, ret, mdd, sharpe = backtest(df)
        results.append((entry, exit_, ret, mdd, sharpe))
        dfs.append(df)
    
    # 打印汇总
    print("\n" + "=" * 60)
    print(" 回测结果汇总")
    print("=" * 60)
    print(f"{'参数(入/出)':<15} {'累计回报':<12} {'最大回撤':<12} {'夏普比率':<10}")
    for (entry, exit_, ret, mdd, sharpe) in results:
        print(f"{entry}/{exit_:<10} {ret:>10.2%} {mdd:>10.2%} {sharpe:>10.4f}")
    
    # 绘图
    plot_full_report(dfs[0], params[0], dfs[1], params[1])