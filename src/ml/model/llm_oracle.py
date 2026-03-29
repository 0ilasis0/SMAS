import hashlib
import json
import sqlite3
import time
from enum import StrEnum

import google.generativeai as genai
import yfinance as yf
from google.api_core.exceptions import ResourceExhausted

from debug import dbg
from path import PathConfig


class TradingMode(StrEnum):
    DAY_TRADE = "day_trade" # 當沖模式：極低延遲，跳過 LLM
    SWING = "swing"         # 波段模式：完整分析，啟用 LLM 與快取


class GeminiOracle:
    """
    量化系統的 LLM 神諭機。
    負責抓取新聞、快取 Payload Hash、並具備自動降級 (Fallback) 的模型切換機制。
    """
    # 定義模型備援清單 (依優先級排序)
    FALLBACK_MODELS = [
        'gemini-3-flash',         # 5 RPM / 20 RPD。
        'gemini-3.1-flash-lite',  # 15 RPM / 500 RPD。
        'gemini-2.5-flash',       # 5 RPM / 20 RPD。
        'gemini-2.5-flash-lite',  # 10 RPM / 20 RPD。
    ]

    def __init__(self, api_keys: list[str], mode: TradingMode = TradingMode.SWING):
        if not api_keys:
            raise ValueError("必須提供至少一把 API Key！")

        self.api_keys = api_keys
        self.mode = mode
        self.db_path = PathConfig.LLM_CACHE

        self.current_key_idx = 0

        self._init_db()

    def _init_db(self):
        """建立 Hash Cache 資料表"""
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
        return hashlib.md5(payload.encode('utf-8')).hexdigest()

    def fetch_recent_news(self, ticker: str) -> str:
        """從 Yahoo Finance 抓取近期新聞並合併為純文本"""
        try:
            stock = yf.Ticker(ticker)
            news_list = stock.news

            if not news_list:
                return ""

            summaries = []
            for item in news_list[:5]:
                title = item.get('title', '')
                publisher = item.get('publisher', '')
                summaries.append(f"【{publisher}】{title}")

            return "\n".join(summaries)
        except Exception as e:
            dbg.war(f"抓取 {ticker} 新聞失敗: {e}")
            return ""

    # 實作模型自動切換引擎
    def _call_gemini_with_fallback(self, prompt: str) -> dict | None:
        num_keys = len(self.api_keys)

        for model_name in self.FALLBACK_MODELS:
            for i in range(num_keys):
                # 動態計算當前的 Index
                attempt_idx = (self.current_key_idx + i) % num_keys
                current_key = self.api_keys[attempt_idx]

                dbg.log(f"嘗試使用模型: {model_name} (API Key: #{attempt_idx + 1})...")

                try:
                    genai.configure(api_key=current_key)
                    model = genai.GenerativeModel(model_name)

                    response = model.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(
                            response_mime_type="application/json",
                            temperature=0.0,
                        )
                    )

                    raw_text = response.text.strip()
                    if raw_text.startswith("```"):
                        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

                    result = json.loads(raw_text)

                    self.current_key_idx = attempt_idx
                    return result

                except ResourceExhausted:
                    dbg.war(f"[{model_name}] API Key #{attempt_idx + 1} 額度耗盡 (429)，切換下一把 Key...")
                    time.sleep(0.5)
                    continue

                except ValueError as ve:
                    dbg.war(f"[{model_name}] API Key #{attempt_idx + 1} 拒絕回答 (安全過濾): {ve}")
                    break

                except json.JSONDecodeError:
                    dbg.error(f"[{model_name}] API Key #{attempt_idx + 1} 輸出非合法 JSON 格式。")
                    break

                except Exception as e:
                    dbg.error(f"[{model_name}] API Key #{attempt_idx + 1} 發生未預期錯誤: {e}")
                    continue

        dbg.error("🚨 嚴重警告：所有模型與所有 API Key 皆已耗盡配額或發生崩潰！")
        return None

    def get_sentiment_score(self, ticker: str) -> tuple[int, str]:
        """
        獲取 1-10 分的情緒分數。
        支援當沖模式 (跳過 LLM) 與波段模式 (快取 + 防幻覺 Prompt)。
        """
        # 當沖模式直接 Bypass，確保零延遲
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

        dbg.log(f"[{ticker}] 未命中快取，啟動 LLM 情緒分析管線...")

        # 嚴格的防幻覺與角色注入 Prompt
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

        # 寫入快取
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sentiment_cache (payload_hash, ticker, score, reason) VALUES (?, ?, ?, ?)",
                (payload_hash, ticker, score, reason)
            )
            conn.commit()

        dbg.log(f"[{ticker}] Gemini 情緒分析完成！分數: {score}, 理由: {reason}")
        return score, reason
