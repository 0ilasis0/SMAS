from bt.actions import (ExecuteBuyNode, ExecuteHoldNode, ExecuteSellNode,
                        GenerateGeminiReportNode, IgnoreFailure)
from bt.conditions import (CheckBuySignalNode, CheckEntryCountLimitNode,
                           CheckHasPositionNode, CheckNotPartialTakenNode,
                           CheckSellSignalNode, CheckStopLossNode,
                           CheckTakeProfitNode, CheckTrailingStopNode)
from bt.core import Selector, Sequence


def build_trading_tree() -> Selector:
    """
    建構 IDSS 量化交易行為樹主邏輯。
    加入「動態加碼 (Scaling In)」與「分級減碼 (Scaling Out)」的機構級策略。
    """

    # ==========================================
    # 策略 1：防守與獲利了結 (優先級最高)
    # ==========================================
    defense_strategy = Sequence("絕對防禦", [
        CheckHasPositionNode(),

        # 依照危機的嚴重程度，拆分成不同級別的出場手段
        Selector("分級出場邏輯", [

            # 級別 1：致命危機 -> 觸發停損或移動停損，100% 資金全面砍倉
            Sequence("強制停損_全面撤退", [
                Selector("停損條件", [
                    CheckStopLossNode(loss_tolerance=-0.05),
                    CheckTrailingStopNode(drawdown_tolerance=-0.08)
                ]),
                ExecuteSellNode(position_ratio=1.0),
                IgnoreFailure(GenerateGeminiReportNode())
            ]),

            # 級別 2：AI 預警 -> 勝率低於 40%，先減碼 50% 降風險，避免被洗出場
            Sequence("勝率低迷_戰術減碼", [
                CheckSellSignalNode(threshold=0.40),
                ExecuteSellNode(position_ratio=0.5),
                IgnoreFailure(GenerateGeminiReportNode())
            ]),

            # 級別 3：極端停利 -> 暴漲 30%，先賣 50% 入袋為安，剩下讓利潤奔跑
            Sequence("極端獲利_部分停利", [
                CheckTakeProfitNode(profit_target=0.30),
                CheckNotPartialTakenNode(),  # 🛡️ 關鍵：已經賣過一半就不會再進來了！
                ExecuteSellNode(position_ratio=0.5),
                IgnoreFailure(GenerateGeminiReportNode())
            ])
        ])
    ])

    # ==========================================
    # 策略 2：進攻與建倉 (允許連續加碼)
    # ==========================================
    attack_strategy = Selector("進攻策略", [

        # 狀況 A：極度看漲 -> 動用當下剩餘現金的 100% 買進
        Sequence("強烈買進", [
            CheckBuySignalNode(threshold=0.75),
            CheckEntryCountLimitNode(max_entries=3),
            ExecuteBuyNode(capital_ratio=1.0),
            IgnoreFailure(GenerateGeminiReportNode())
        ]),

        # 狀況 B：普通看漲 -> 動用當下剩餘現金的 50% 試水溫或微調加碼
        Sequence("保守買進", [
            # (已移除空手限制)
            CheckBuySignalNode(threshold=0.60),
            CheckEntryCountLimitNode(max_entries=3),
            ExecuteBuyNode(capital_ratio=0.5),
            IgnoreFailure(GenerateGeminiReportNode())
        ])
    ])

    # ==========================================
    # 策略 3：觀望發呆 (條件都不滿足時的預設動作)
    # ==========================================
    hold_strategy = Sequence("觀望策略", [
        ExecuteHoldNode(),
        # IgnoreFailure(GenerateGeminiReportNode())
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
