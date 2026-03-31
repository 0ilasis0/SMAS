# ui/stock_names.py
import requests
import streamlit as st

from debug import dbg


@st.cache_data(ttl=86400)  # 設定快取壽命為 86400 秒 (1天)，每天自動更新一次即可
def get_tw_stock_mapping() -> dict:
    """
    從台灣證交所與櫃買中心 OpenAPI 動態抓取所有股票代號與中文名稱。
    回傳格式: {"2330.TW": "台積電", "5469.TW": "瀚宇博", "2337.TW": "旺宏", ...}
    """
    mapping = {}

    # 1. 抓取上市股票 (TWSE)
    try:
        twse_url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        res = requests.get(twse_url, timeout=5)
        if res.status_code == 200:
            for item in res.json():
                # 組合出 Yahoo Finance 格式的代號 (例如 2330.TW)
                mapping[f"{item['Code']}.TW"] = item['Name']
    except Exception as e:
        dbg.war(f"取得上市股票名稱失敗: {e}")

    # 2. 抓取上櫃股票 (TPEx)
    try:
        tpex_url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
        res = requests.get(tpex_url, timeout=5)
        if res.status_code == 200:
            for item in res.json():
                # 上櫃股票在 Yahoo Finance 是 .TWO
                mapping[f"{item['SecuritiesCompanyCode']}.TWO"] = item['CompanyName']
    except Exception as e:
        dbg.war(f"取得上櫃股票名稱失敗: {e}")

    # 如果 API 剛好掛掉，給幾個預設值當作防呆底線
    if not mapping:
        return {"2330.TW": "台積電", "2337.TW": "旺宏", "5469.TW": "瀚宇博", "2388.TW": "威盛"}

    return mapping