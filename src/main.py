from base import central_mg
from data.fetcher import Fetcher
from data.manager import DataManager
from data.variable import TimeUnit

if __name__ == "__main__":
    ticker = "0052.TW"

    # 實例化兩個核心組件
    fetcher = Fetcher()
    db = DataManager()

    # 1. 測試自選股清單
    db.add_to_watchlist(ticker)

    # 2. 抓取並儲存日 K 線 (波段模式)
    daily_df = fetcher.fetch_daily_data(ticker, 1, TimeUnit.MONTH)
    db.save_daily_data(ticker, daily_df)

    # 3. 抓取並儲存分時 K 線 (當沖模式)
    intraday_df = fetcher.fetch_intraday_data(ticker, 5)
    db.save_intraday_data(ticker, intraday_df)



while (central_mg.running):
    pass
