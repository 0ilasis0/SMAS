from bt.actions import ExecuteBuyNode, ExecuteHoldNode, ExecuteSellNode
from bt.conditions import (CheckBuySignalNode, CheckCooldownNode,
                           CheckEntryCountLimitNode, CheckGapLimitNode,
                           CheckHasPositionNode, CheckNotOverheatedNode,
                           CheckNotPartialTakenNode,
                           CheckSellSentimentFilterNode, CheckSellSignalNode,
                           CheckSentimentFilterNode, CheckStopLossNode,
                           CheckTakeProfitNode, CheckTrailingStopNode,
                           CheckTrendFilterNode)
from bt.core import Inverter, Selector, Sequence
from bt.strategy_config import StrategyConfig


def build_trading_tree(config: StrategyConfig) -> Selector:
    """
    建構 IDSS 量化交易行為樹主邏輯。
    :param config: 策略參數設定，若不傳入則使用預設值。
    """

    # ==========================================
    # 策略 1：防守與獲利了結 (優先級最高)
    # ==========================================
    ai_sell_conditions = []
    if config.enable_llm_oracle:
        ai_sell_conditions.append(CheckSellSentimentFilterNode(block_score=config.block_sell_sentiment_score))

    defense_strategy = Sequence("絕對防禦", [
        CheckHasPositionNode(),

        Selector("分級出場邏輯", [

            # 級別 1：致命危機 -> 觸發停損或移動停損，全面砍倉
            Sequence("強制停損_全面撤退", [
                Selector("停損條件", [
                    CheckStopLossNode(loss_tolerance=config.stop_loss_tolerance, cooldown_days=config.cooldown_days),
                    CheckTrailingStopNode(drawdown_tolerance=config.trailing_stop_drawdown, cooldown_days=config.cooldown_days)
                ]),
                ExecuteSellNode(position_ratio=config.stop_loss_sell_ratio),
                # ForceSuccess(GenerateGeminiReportNode())
            ]),

            # 級別 2：AI 預警 -> 勝率低迷，先減碼降風險
            Sequence("勝率低迷_戰術減碼", [
                CheckSellSignalNode(threshold=config.sell_signal_threshold, sell_risk=config.sell_risk)] +
                ai_sell_conditions +
                [ExecuteSellNode(position_ratio=config.warning_sell_ratio),
                # ForceSuccess(GenerateGeminiReportNode())
            ]),

            # 級別 3：極端停利 -> 暴漲達標，先入袋為安，剩下讓利潤奔跑
            Sequence("極端獲利_部分停利", [
                CheckTakeProfitNode(profit_target=config.take_profit_target),
                CheckNotPartialTakenNode(),
                Inverter("非強烈看漲", CheckBuySignalNode(threshold=config.strong_buy_threshold, buy_risk=config.buy_risk)),
                ExecuteSellNode(position_ratio=config.take_profit_sell_ratio),
                # ForceSuccess(GenerateGeminiReportNode())
            ])
        ])
    ])

    # ==========================================
    # 策略 2：進攻與建倉 (允許連續加碼)
    # ==========================================
    attack_conditions = [
        CheckCooldownNode(cooldown_days=config.cooldown_days),
        CheckTrendFilterNode(safe_threshold=config.safe_threshold),
        CheckNotOverheatedNode(max_return_5d=config.max_return_5d, max_bias_20=config.max_bias_20)
    ]
    if config.enable_llm_oracle:
        attack_conditions.append(CheckSentimentFilterNode(min_score=config.min_sentiment_score))

    attack_strategy = Sequence("進攻策略大門", attack_conditions + [
        # 通過防禦後，才進入選擇器分配力道
        Selector("買進力道分配", [

            # 狀況 A：極度看漲 -> 強烈買進
            Sequence("強烈買進", [
                CheckBuySignalNode(threshold=config.strong_buy_threshold, buy_risk=config.buy_risk),
                CheckEntryCountLimitNode(max_entries=config.max_entries),
                CheckGapLimitNode(max_gap_ratio=config.max_gap_ratio),
                ExecuteBuyNode(capital_ratio=config.strong_buy_capital_ratio),
                # ForceSuccess(GenerateGeminiReportNode())
            ]),

            # 狀況 B：普通看漲 -> 保守買進試水溫
            Sequence("保守買進", [
                CheckBuySignalNode(threshold=config.conservative_buy_threshold, buy_risk=config.buy_risk),
                CheckEntryCountLimitNode(max_entries=config.max_entries),
                CheckGapLimitNode(max_gap_ratio=config.max_gap_ratio),
                ExecuteBuyNode(capital_ratio=config.conservative_buy_capital_ratio),
                # ForceSuccess(GenerateGeminiReportNode())
            ])
        ])
    ])

    # ==========================================
    # 策略 3：觀望發呆 (條件都不滿足時的預設動作)
    # ==========================================
    hold_strategy = Sequence("觀望策略", [
        ExecuteHoldNode(),
        # ForceSuccess(GenerateGeminiReportNode())
    ])

    # ==========================================
    # 總指揮：Root 節點
    # ==========================================
    root = Selector("主交易邏輯", [
        defense_strategy,
        attack_strategy,
        hold_strategy
    ])

    return root