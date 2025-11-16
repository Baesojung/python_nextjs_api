from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ETFConfig:
    """ETF 설정값 (Next.js/백테스트 공용)."""

    label: str
    yahoo_symbol: str
    region: str


# NOTE: 실제 서비스에 맞게 ETF 목록을 확장하세요.
ETF_LIST: List[ETFConfig] = [
    ETFConfig(label="VOO", yahoo_symbol="VOO", region="US"),
    ETFConfig(label="QQQ", yahoo_symbol="QQQ", region="US"),
    ETFConfig(label="SPY", yahoo_symbol="SPY", region="US"),
    ETFConfig(label="KODEX 200", yahoo_symbol="069500.KS", region="KR"),
    ETFConfig(label="TIGER 200", yahoo_symbol="102110.KS", region="KR"),
    ETFConfig(label="TQQQ", yahoo_symbol="TQQQ", region="US"),
    ETFConfig(label="UPRO", yahoo_symbol="UPRO", region="US"),
]


TAX_RULES: Dict[str, Dict[str, float]] = {
    "US": {
        "transaction_cost": 0.001,  # 편도 0.1%
        "dividend_tax": 0.154,  # 15.4%
    },
    "KR": {
        "transaction_cost": 0.0005,  # 편도 0.05%
        "dividend_tax": 0.15,  # 15%
    },
}
