from pybit.unified_trading import HTTP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

session = HTTP(
    testnet=False,  # ← 真實主網
    api_key="你的KEY",#MFQttzNrslm5TPOu0B
    api_secret="你的SECRET",#oQY4GHFgPJ6QSdw2zeHfG0U49Pnt5BAvWHLy
)

def get_klines(symbol, interval, limit):
    resp = session.get_kline(
        category="linear",      # 永續
        symbol=symbol,
        interval=str(interval),
        limit=limit,
    )

    raw = resp["result"]["list"]

    df = pd.DataFrame(raw, columns=[
        "timestamp","open","high","low","close","volume","turnover"
    ])

    df = df.iloc[::-1].reset_index(drop=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")
    df[["open","high","low","close","volume","turnover"]] = df[
        ["open","high","low","close","volume","turnover"]
    ].astype(float)

    return df
def find_previous_klines(symbol, start_date, end_date):
    """
    抓特定日期範圍的日 K（interval = D）
    """
    start_ms = int(datetime.datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ms   = int(datetime.datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    all_frames = []
    limit = 1000
    cur_start = start_ms

    while True:
        resp = session.get_kline(
            category="linear",
            symbol=symbol,
            interval="D",
            limit=limit,
            start=cur_start,
            end=end_ms
        )

        data = resp["result"]["list"]
        if not data:
            break

        df = pd.DataFrame(data, columns=[
            "timestamp","open","high","low","close","volume","turnover"
        ])
        all_frames.append(df)

        # 找到這批資料中最早一筆 → 當成下一輪的 start
        earliest = int(df["timestamp"].min())
        cur_start = earliest + 1

        # 如果資料少於 1000 筆，代表抓完了
        if len(data) < limit:
            break

        time.sleep(0.2)  # 避免被 rate-limit

    result = pd.concat(all_frames, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], unit="ms")
    result = result.sort_values("timestamp").reset_index(drop=True)
    result[["open","high","low","close","volume","turnover"]] = result[
        ["open","high","low","close","volume","turnover"]
    ].astype(float)

    return result
def add_bollinger_bands(df, window, num_std):
    """
    在 df 上加上布林通道：
    - middle_band: 中軌 = 收盤價 SMA
    - upper_band: 上軌 = SMA + k*STD
    - lower_band: 下軌 = SMA - k*STD
    """
    df = df.copy()
    df["middle_band"] = df["close"].rolling(window).mean()
    df["std"] = df["close"].rolling(window).std()
    df["upper_band"] = df["middle_band"] + num_std * df["std"]
    df["lower_band"] = df["middle_band"] - num_std * df["std"]
    df["bandwidth"] = (df["upper_band"] - df["lower_band"]) / df["middle_band"]
    return df
def classify_regime(
    df,
    squeeze_bw,   # 布林帶寬 < 5% 視為擠壓
    trend_bw,     # 布林帶寬 > 12% 且斜率大視為趨勢
    trend_slope# 中軌斜率門檻
):
    """
    新增一欄 'regime'：'squeeze', 'range', 'trend'
    """
    df = df.copy()

    # 中軌的斜率（看 MA 方向） - 用 N 根差分平滑一下
    slope_window = 5
    df["ma_slope"] = (
        df["middle_band"].diff(slope_window) /
        slope_window /
        df["middle_band"]
    )

    # 擠壓：布林帶很窄
    squeeze_mask = df["bandwidth"] < squeeze_bw

    # 趨勢：布林很寬 + MA 有明顯方向
    trend_mask = (df["bandwidth"] > trend_bw) & (df["ma_slope"].abs() > trend_slope)

    # 其他就是盤整
    regime = np.where(
        squeeze_mask, "squeeze",
        np.where(trend_mask, "trend", "range")
    )
    df["regime"] = regime

    return df
def generate_signals(df):
    """
    三種策略整合：
    - range: 布林上下軌做均值回歸（反向）
    - trend: 突破上/下軌做順勢
    - squeeze: 當成趨勢前兆，突破時才進場
    """
    df = df.copy()
    df["signal"] = 0     # 當根的操作：1 開多，-1 開空，0 無動作/平倉
    df["position"] = 0   # 當根收盤持倉：1 多頭，-1 空頭，0 空手

    pos = 0  # 當前部位

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        regime = row["regime"]
        sig = 0

        # ======= 沒持倉時：找進場點 =======
        if pos == 0:
            if regime == "range":
                # 盤整 → 均值回歸
                if row["close"] < row["lower_band"]:
                    # 跌破下軌 → 反向作多
                    sig = 1
                    pos = 1
                elif row["close"] > row["upper_band"]:
                    # 突破上軌 → 反向作空
                    sig = -1
                    pos = -1

            elif regime in ["trend", "squeeze"]:
                # 趨勢 / 擠壓 → 突破順勢
                widening = row["bandwidth"] > prev["bandwidth"]
                if (
                    row["close"] > row["upper_band"]
                    and prev["close"] <= prev["upper_band"]
                    and widening
                ):
                    # 價格向上突破 + 布林正在張開 → 做多
                    sig = 1
                    pos = 1
                elif (
                    row["close"] < row["lower_band"]
                    and prev["close"] >= prev["lower_band"]
                    and widening
                ):
                    # 價格向下突破 + 布林正在張開 → 做空
                    sig = -1
                    pos = -1

        # ======= 有持倉時：找出場點 =======
        else:
            if pos == 1:  # 手上有多單
                if regime == "range":
                    # 盤整 → 回到中軌或以上就平倉
                    if row["close"] >= row["middle_band"]:
                        sig = 0
                        pos = 0
                elif regime in ["trend", "squeeze"]:
                    # 趨勢 → 跌破中軌視為趨勢結束
                    if row["close"] < row["middle_band"]:
                        sig = 0
                        pos = 0

            elif pos == -1:  # 手上有空單
                if regime == "range":
                    if row["close"] <= row["middle_band"]:
                        sig = 0
                        pos = 0
                elif regime in ["trend", "squeeze"]:
                    if row["close"] > row["middle_band"]:
                        sig = 0
                        pos = 0

        df.loc[i, "signal"] = sig
        df.loc[i, "position"] = pos

    return df
def backtest(df, periods_per_year):
    """
    用收盤價做極簡回測：
    - 不含手續費與滑價
    - position shift(1) 模擬下一根才成交
    - 回傳：df, 總報酬, 夏普比率, 最大回撤
    """
    df = df.copy()
    # 單期報酬（這裡是一根K線的報酬）
    df["ret"] = df["close"].pct_change()
    df["strategy_ret"] = df["position"].shift(1) * df["ret"]

    # 策略權益曲線
    df["equity"] = (1 + df["strategy_ret"].fillna(0)).cumprod()

    # 總報酬（從1變成多少）
    total_return = df["equity"].iloc[-1] 

    # 夏普比率（假設無風險利率=0）
    # 年化：日K用 252；如果是1h K線，改成 24*365 ≈ 8760
    mean_ret = df["strategy_ret"].mean()
    std_ret = df["strategy_ret"].std()
    if std_ret == 0 or pd.isna(std_ret):
        sharpe = 0.0
    else:
        sharpe = (mean_ret * periods_per_year) / (std_ret * np.sqrt(periods_per_year))

    # 最大回撤
    equity = df["equity"]
    cum_max = equity.cummax()
    drawdown = equity / cum_max - 1
    max_drawdown = drawdown.min()  # 會是負數，例如 -0.35 = 回撤35%

    print(f"策略總報酬：約 {total_return * 100:.2f}%")
    print(f"夏普比率：約 {sharpe:.2f}")
    print(f"最大回撤：約 {max_drawdown * 100:.2f}%")

    # 把 drawdown 也存到 df，等等畫圖用
    df["drawdown"] = drawdown

    return df, total_return, sharpe, max_drawdown
def plot_performance(df, total_return, sharpe, max_drawdown):

    """
    畫出：
    - 策略權益曲線
    - 下方陰影顯示回撤
    - 左上角顯示 總收益 / 夏普 / 最大回撤
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 權益曲線
    ax1.plot(df["timestamp"], df["equity"], label="Equity Curve", linewidth=1.5)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Equity")
    ax1.grid(True)

    # 第二個 y 軸畫回撤
    ax2 = ax1.twinx()
    ax2.fill_between(df["timestamp"], df["drawdown"], 0,
                     alpha=0.3, label="Drawdown")
    ax2.set_ylabel("Drawdown")

    # 把績效指標寫在圖上（左上角）
    textstr = (
        f"Total Return: {total_return * 100:.2f}%\n"
        f"Sharpe: {sharpe:.2f}\n"
        f"Max DD: {max_drawdown * 100:.2f}%"
    )
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round", alpha=0.3))

    # 圖例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.title("Strategy Performance")
    plt.tight_layout()
    plt.show()
def plot_price_with_trades(df):
    """
    畫出：
    - 價格 close
    - 布林三條線
    - 根據 position 變化標出進出場點
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # 價格 & 布林帶
    ax.plot(df["timestamp"], df["close"], label="Close", linewidth=1.2)
    ax.plot(df["timestamp"], df["upper_band"], label="Upper Band", linewidth=1)
    ax.plot(df["timestamp"], df["middle_band"], label="Middle Band", linewidth=1)
    ax.plot(df["timestamp"], df["lower_band"], label="Lower Band", linewidth=1)

    # 用 position 的變化來找「進場 / 出場」點
    pos = df["position"].fillna(0)

    long_entry  = (pos.shift(1) == 0) & (pos == 1)
    long_exit   = (pos.shift(1) == 1) & (pos == 0)
    short_entry = (pos.shift(1) == 0) & (pos == -1)
    short_exit  = (pos.shift(1) == -1) & (pos == 0)

    # 進出場點畫在價格線上
    ax.scatter(
        df.loc[long_entry, "timestamp"],
        df.loc[long_entry, "close"],
        marker="^", s=80, label="Long Entry"
    )
    ax.scatter(
        df.loc[long_exit, "timestamp"],
        df.loc[long_exit, "close"],
        marker="v", s=80, label="Long Exit"
    )
    ax.scatter(
        df.loc[short_entry, "timestamp"],
        df.loc[short_entry, "close"],
        marker="v", s=80, label="Short Entry"
    )
    ax.scatter(
        df.loc[short_exit, "timestamp"],
        df.loc[short_exit, "close"],
        marker="^", s=80, label="Short Exit"
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USDT)")
    ax.set_title("ETHUSDT Price with Bollinger Bands & Trades")
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

#---------選時間----------
#df_now = get_klines("ETHUSDT", "60", 200)
#df_previous = find_previous_klines("ETHUSDT", "2025-01-01", "2025-11-30")
#-------------------------
#df_booling = add_bollinger_bands(df_previous, window=40, num_std=2.5)

if __name__ == "__main__":
    # 例子：抓 1h ETHUSDT 近 1000 根，套三策略
    df = get_klines("ETHUSDT", "D", 500)
    df_previous = find_previous_klines("ETHUSDT", "2022-01-01", "2022-12-31")
    df_booling = add_bollinger_bands(df_previous, window=15, num_std=2.5)
    df_regime = classify_regime(df_booling,squeeze_bw=0.05,trend_bw=0.12,trend_slope=0.0005)
    df_signal = generate_signals(df_regime)
    df_back, total_return, sharpe, max_drawdown = backtest(df_signal, periods_per_year=500)
    # 畫績效圖
    plot_performance(df_back, total_return, sharpe, max_drawdown)
    plot_price_with_trades(df_back)
