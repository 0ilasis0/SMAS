from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from data.const import StockCol
from debug import dbg

from ._const import DataConst

if TYPE_CHECKING:
    from .core import QuantAIEngine

class DataWatchdog:
    """
    專門負責檢查資料庫股價是否有不合理的斷層 (如除權息、分割)。
    若發現異常，會自動啟動「本地強制平滑演算法」進行修復。
    """
    def __init__(self, engine):
        self.engine: "QuantAIEngine" = engine

    def run_data_watchdog(self, ticker: str):
        """ 撈出資料並檢查是否有除權息斷層，有則強制啟動修復 """
        df_raw = self.engine.db.get_daily_data(ticker)

        needs_healing = self._check_data_integrity(ticker, df_raw)

        if needs_healing:
            dbg.log(f"🛠️ [Watchdog] 準備啟動 {ticker} 的歷史資料平滑修復程序...")
            self._auto_heal_corporate_actions(ticker, df_raw)

    def _check_data_integrity(self, ticker: str, df: pd.DataFrame) -> bool:
        """
        檢查資料庫股價是否有不合理的斷層。
        回傳 True 代表需要進行除權息/分割修復。
        """
        if len(df) < 2: return False

        if StockCol.ADJ_CLOSE not in df.columns:
            dbg.war(f"🚨 [Watchdog] 警告！{ticker} 缺少 {StockCol.ADJ_CLOSE} 欄位，請重新抓取。")
            return False

        df_safe = df[df[StockCol.CLOSE] > 0].copy()
        if len(df_safe) < 2: return False

        # 檢查原始價格，找出單日漲跌幅絕對值大於~的日子 (台股漲跌幅上限為 10%)
        raw_returns = df[StockCol.CLOSE].pct_change().dropna()
        raw_anomaly = raw_returns.abs() > DataConst.RETURNS_ABS

        if raw_anomaly.any():
            dbg.war(f"🚨 [Watchdog] 警告！偵測到 {ticker} 歷史股價發生 >{int(DataConst.RETURNS_ABS * 100)}% 的斷層 (極可能為股票分割)！")
            return True

        return False

    def _auto_heal_corporate_actions(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        自動偵測除權息與股票分割/減資。
        若 Yahoo API 提供的資料未還原，啟動「本地強制平滑演算法 (Backward Adjustment)」手動修復。
        """
        if len(df) < 2:
            return df

        # 找出收盤價異常為 0 的日子，發出警告並將其捨棄 (視同休市)
        zero_mask = df[StockCol.CLOSE] <= 0
        if zero_mask.any():
            zero_dates = df[zero_mask].index.strftime('%Y-%m-%d').tolist()
            dbg.war(f"[Watchdog] 發現 {ticker} 有 {len(zero_dates)} 筆收盤價為 0 的損毀資料 (如: {zero_dates[0]})。已直接捨棄，視同休市處理。")
            df_safe = df[~zero_mask].copy()
        else:
            df_safe = df.copy()

        daily_returns = df_safe[StockCol.CLOSE].pct_change().dropna()
        anomaly_mask = daily_returns.abs() > DataConst.RETURNS_ABS

        if anomaly_mask.any():
            dbg.war(f"🚨 [Watchdog] 偵測到 {ticker} 出現異常跳空 (大於 {int(DataConst.RETURNS_ABS * 100)}%)！")

            self.engine.db.clear_ticker_data(ticker)
            df_healed = self.engine.stock_fetcher.fetch_daily_data(
                ticker,
                period=DataConst.HEAL_PERIOD,
                unit=DataConst.HEAL_UNIT
            )

            if not df_healed.empty:
                # 重抓後依然要過濾零值
                new_zero_mask = df_healed[StockCol.CLOSE] <= 0
                if new_zero_mask.any():
                    df_healed = df_healed[~new_zero_mask].copy()

                new_returns = df_healed[StockCol.CLOSE].pct_change().dropna()
                new_anomaly_mask = new_returns.abs() > DataConst.RETURNS_ABS

                if new_anomaly_mask.any():
                    dbg.war(f"[Watchdog] Yahoo 源頭資料依然損毀！啟動「本地端強制平滑修復」...")

                    anomaly_dates = new_returns[new_anomaly_mask].index.sort_values(ascending=False)

                    # 執行強制平滑演算法
                    for adate in anomaly_dates:
                        idx_loc = df_healed.index.get_loc(adate)

                        if idx_loc > 0:
                            prev_date = df_healed.index[idx_loc - 1]

                            price_after = float(df_healed[StockCol.OPEN].iloc[idx_loc])
                            price_before = float(df_healed[StockCol.CLOSE].iloc[idx_loc - 1])

                            if price_before <= 0 or price_after <= 0:
                                dbg.war(f"[Watchdog] {adate.strftime('%Y-%m-%d')} 股價異常為 0，放棄修復此斷層。")
                                continue

                            ratio = price_after / price_before

                            dbg.log(f"🛠️ [Watchdog] 執行平滑修復作業：")
                            dbg.log(f"   ➤ 斷層前最後交易日: {prev_date.strftime('%Y-%m-%d')} (收盤: {price_before:.2f})")
                            dbg.log(f"   ➤ 斷層生效日: {adate.strftime('%Y-%m-%d')} (開盤: {price_after:.2f})")
                            dbg.log(f"   ➤ 計算還原比例: {ratio:.4f}")

                            if ratio < DataConst.HEAL_MIN_RATIO or ratio > DataConst.HEAL_MAX_RATIO:
                                dbg.war(
                                    f"[Watchdog] {adate.strftime('%Y-%m-%d')} 偵測到不合理的極端斷層 "
                                    f"(ratio={ratio:.4f})，可能為嚴重髒資料，放棄本地平滑修復。"
                                )
                                continue

                            price_cols = [StockCol.OPEN, StockCol.HIGH, StockCol.LOW, StockCol.CLOSE]
                            df_healed.loc[:prev_date, price_cols] *= ratio

                            new_vol = df_healed.loc[:prev_date, StockCol.VOLUME] / ratio
                            new_vol.replace([np.inf, -np.inf], np.nan, inplace=True)
                            new_vol.fillna(0, inplace=True)
                            df_healed.loc[:prev_date, StockCol.VOLUME] = new_vol.round().astype('int64')

                    dbg.log(f"✅ [Watchdog] {ticker} 本地強制還原修復完成！所有技術指標將恢復正常。")
                else:
                    dbg.log(f"✅ [Watchdog] {ticker} Yahoo 資料自動還原成功！")

                self.engine.db.save_daily_data(ticker, df_healed)
                return df_healed
            else:
                dbg.error(f"❌ [Watchdog] 修復失敗：無法重新抓取資料。")
                return df_safe

        # 如果本來就沒斷層，也要把剔除過 0 值的乾淨資料存回去
        self.engine.db.save_daily_data(ticker, df_safe)
        return df_safe