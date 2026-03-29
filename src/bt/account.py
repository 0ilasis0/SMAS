from dataclasses import dataclass


@dataclass
class Account:
    """
    實體帳戶/資金池。
    在多檔股票同時運行時，所有股票的黑板都必須指向同一個 Account 實體。
    """
    cash: int = 0
