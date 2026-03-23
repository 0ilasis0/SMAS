from debug import dbg


class FrozenMeta(type):
    def __setattr__(cls, name, value):
        # 攔截對類別屬性的賦值行為
        raise AttributeError(f"無法修改常數屬性: '{name}'")



class MathTool:
    @staticmethod
    def clamp(val: int, min_val: int, max_val: int):
        if max_val < min_val:
            dbg.war("clamp 參數顛倒，已自動交換 min/max")
            min_val, max_val = max_val, min_val
        return max(min(val, max_val), min_val)



#
# 狀態總管理
#
class CentralManager:
    def __init__(self):
        self.running: bool = False

central_mg = CentralManager()
