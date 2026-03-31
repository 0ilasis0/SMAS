import hashlib
import json
import sqlite3
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from enum import StrEnum

from google import genai
from google.genai import types
from google.genai.errors import APIError

from debug import dbg
from path import PathConfig
from ui.const import EncodingConst


class TradingMode(StrEnum):
    DAY_TRADE = "day_trade"
    SWING = "swing"

class GeminiOracle:
    """
    量化系統的 LLM 。
    實作完美隔離的 API Key Client 切換機制。
    """
    FALLBACK_MODELS = [
        'models/gemini-3.1-flash-lite-preview', # 🥇 (15 RPM / 500 RPD) - 海量掃描專用
        'models/gemini-3-flash-preview',        # 🥈 (5 RPM / 20 RPD) - 高智商，專解複雜新聞
        'models/gemini-2.5-flash-lite',         # 🥉 (10 RPM / 20 RPD) - 伺服器波動時替補
        'models/gemini-2.5-flash',              # 🎖️ (5 RPM / 20 RPD) - 最後底線
    ]

    def __init__(self, api_keys: list[str], mode: TradingMode = TradingMode.SWING):
        if not api_keys:
            raise ValueError("必須提供至少一把 API Key！")

        self.api_keys = api_keys
        self.mode = mode
        self.db_path = PathConfig.LLM_CACHE
        self._init_db()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_cache (
                    payload_hash TEXT PRIMARY KEY,
                    ticker TEXT,
                    score INTEGER,
                    reason TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def _get_payload_hash(self, ticker: str, news_text: str) -> str:
        payload = f"{ticker}_{news_text}"
        return hashlib.md5(payload.encode(EncodingConst.UTF8)).hexdigest()

    def fetch_recent_news(self, ticker: str) -> str:
        """
        透過 Google News RSS 抓取台股在地化繁體中文新聞。
        比 yfinance 穩定 100 倍，且精準度極高。
        """
        try:
            # 確保關鍵字精準：例如把 "2330.TW" 變成 "2330 股票"
            search_keyword = ticker.replace('.TW', '').replace('.TWO', '') + " 股票"

            # 將中文與符號進行 URL 編碼
            query = urllib.parse.quote(search_keyword)

            # 組裝 Google News RSS 網址 (指定台灣地區、繁體中文)
            url = f"https://news.google.com/rss/search?q={query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"

            # 偽裝成正常瀏覽器發送請求，避免被擋
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                xml_data = response.read()

            # 解析 XML 資料
            root = ET.fromstring(xml_data)
            items = root.findall('.//item')

            if not items:
                dbg.war(f"[{ticker}] Google 新聞查無資料")
                return ""

            # 萃取前 5 則最新新聞標題與來源
            summaries = []
            for item in items[:5]:
                title = item.find('title').text if item.find('title') is not None else ""
                pub_date = item.find('pubDate').text if item.find('pubDate') is not None else "未知時間"

                summaries.append(f"[{pub_date}] 【新聞】{title}")

            return "\n".join(summaries)

        except Exception as e:
            dbg.war(f"抓取 {ticker} 新聞失敗: {e}")
            return ""

    def _call_gemini_with_fallback(self, prompt: str) -> dict | None:
        """嘗試以二維瀑布機制呼叫 Gemini (新版 SDK 實作)"""

        for model_name in self.FALLBACK_MODELS:
            for key_idx, current_key in enumerate(self.api_keys):
                dbg.log(f"嘗試使用模型: {model_name} (API Key: #{key_idx + 1})...")

                try:
                    client = genai.Client(api_key=current_key)

                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.0,
                        )
                    )

                    raw_text = response.text.strip()
                    if "```" in raw_text:
                        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

                    # 避免 LLM 廢話開頭，直接找第一個 { 和最後一個 }
                    start_idx = raw_text.find('{')
                    end_idx = raw_text.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        raw_text = raw_text[start_idx:end_idx+1]

                    return json.loads(raw_text)

                except APIError as api_err:
                    if api_err.code == 429:
                        dbg.war(f"[{model_name}] API Key #{key_idx + 1} 額度耗盡 (429)，切換下一把 Key...")
                        time.sleep(0.5)
                        continue
                    else:
                        dbg.error(f"[{model_name}] API Key #{key_idx + 1} 發生 API 錯誤 (Code: {api_err.code}): {api_err.message}")
                        break

                except ValueError as ve:
                    dbg.war(f"[{model_name}] API Key #{key_idx + 1} 拒絕回答 (安全過濾): {ve}")
                    break

                except json.JSONDecodeError:
                    dbg.error(f"[{model_name}] API Key #{key_idx + 1} 輸出非合法 JSON 格式。")
                    break

                except Exception as e:
                    dbg.error(f"[{model_name}] API Key #{key_idx + 1} 發生未預期錯誤: {e}")
                    continue

        dbg.error("🚨 嚴重警告：所有模型與所有 API Key 皆已耗盡配額或發生崩潰！")
        return None

    def get_sentiment_score(self, ticker: str) -> tuple[int, str]:
        if self.mode == TradingMode.DAY_TRADE:
            dbg.log(f"[{ticker}] 當沖模式啟動：略過 LLM 判斷，給予中立分數 5")
            return 5, "當沖模式，略過新聞分析。"

        news_text = self.fetch_recent_news(ticker)

        if not news_text.strip():
            return 5, "無最新重大新聞。"

        payload_hash = self._get_payload_hash(ticker, news_text)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT score, reason FROM sentiment_cache WHERE payload_hash=?", (payload_hash,))
            row = cursor.fetchone()
            if row:
                dbg.log(f"[{ticker}] 命中 LLM 快取！直接取回情緒分數: {row[0]}")
                return row[0], row[1]

        dbg.log(f"[{ticker}] 未命中快取，啟動 LLM 多線程情緒分析管線...")

        prompt = f"""
        你是一個沒有任何主觀情緒、只依據事實進行量化標記的交易演算法元件。
        請閱讀以下關於 {ticker} 的近期新聞標題，並給出一個 1 到 10 分的情緒分數。

        【評分絕對基準】：
        1-3 分：重大利空 (如財報暴雷、高層舞弊、掉單、大環境崩盤)。
        4-6 分：中立或雜訊 (如常規法說會預告、無關緊要的產業新聞)。
        7-10 分：重大利多 (如財報超預期、接獲大單、併購、政策大幅利多)。

        【近期新聞】：
        {news_text}

        【嚴格輸出限制】：
        你被禁止輸出任何解釋性文字、Markdown 符號或警告語語。你只能輸出純 JSON。
        格式必須完全符合：
        {{"score": 整數, "reason": "15字以內客觀事實陳述"}}
        """

        result = self._call_gemini_with_fallback(prompt)

        if result is None:
            return 5, "API 額度耗盡或全面癱瘓，給予中立保護分數。"

        score = int(result.get('score', 5))
        reason = result.get('reason', '無')

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sentiment_cache (payload_hash, ticker, score, reason) VALUES (?, ?, ?, ?)",
                (payload_hash, ticker, score, reason)
            )
            conn.commit()

        dbg.log(f"[{ticker}] Gemini 情緒分析完成！分數: {score}, 理由: {reason}")
        return score, reason
