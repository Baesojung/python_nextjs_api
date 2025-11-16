import datetime as dt
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from etfBackTesting import ETFConfig, ETF_LIST, TAX_RULES

DAILY_CONTRIBUTION = 5_000  # 원화 기준 일별 납입금
DEFAULT_LOOKBACK_DAYS = 365


class BacktestRequest(BaseModel):
    """Next.js에서 전달받을 요청 본문 정의."""

    start_date: Optional[dt.date] = Field(
        None, description="백테스트 시작일 (미입력 시 1년 전)", example="2023-01-01"
    )
    end_date: Optional[dt.date] = Field(
        None, description="백테스트 종료일 (미입력 시 오늘)", example="2024-01-01"
    )
    daily_contribution: float = Field(
        DAILY_CONTRIBUTION, gt=0, description="거래일당 납입액 (원화)"
    )
    symbols: Optional[List[str]] = Field(
        None,
        description="분석할 ETF 라벨 또는 야후 티커 목록. 미입력 시 전체 ETF_LIST 사용",
        example=["VOO", "QQQ"],
    )

    @validator("end_date")
    def validate_dates(
        cls, end_date: Optional[dt.date], values: Dict[str, Any]
    ) -> Optional[dt.date]:
        start_date: Optional[dt.date] = values.get("start_date")
        if start_date and end_date and start_date > end_date:
            raise ValueError("start_date must be before end_date")
        return end_date


class BacktestResponse(BaseModel):
    start_date: dt.date
    end_date: dt.date
    daily_contribution: float
    etf_count: int
    results: List[Dict[str, Any]]
    errors: List[str]


router = APIRouter(prefix="/api", tags=["ETF Periodic Investment"])


app = FastAPI(title="ETF Periodic Investment API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


def _to_date(value) -> dt.date:
    """다양한 날짜 표현을 datetime.date로 통일."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return pd.Timestamp(value).date()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"날짜로 변환할 수 없습니다: {value}") from exc


def _load_history(symbol: str, start: dt.date, end: dt.date) -> pd.Series:
    """요청 기간의 수정주가를 다운로드."""
    yf_end = end + dt.timedelta(days=1)
    data = yf.download(symbol, start=start, end=yf_end, auto_adjust=False, progress=False)
    if data.empty or "Adj Close" not in data:
        raise RuntimeError(f"{symbol} 가격 데이터를 받지 못했습니다.")
    prices = data["Adj Close"].dropna()
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze("columns")
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    if prices.empty:
        raise RuntimeError(f"{symbol} 수정주가 시계열이 비어 있습니다.")
    return prices


def _load_dividend_series(symbol: str, start: dt.date, end: dt.date) -> pd.Series:
    """기간 내 배당 시계열을 반환."""
    ticker = yf.Ticker(symbol)
    dividends = ticker.dividends
    if dividends.empty:
        return pd.Series(dtype=float)
    mask = (dividends.index.date >= start) & (dividends.index.date <= end)
    return dividends.loc[mask]


def _resolve_configs(symbols: Optional[Sequence[str]]) -> Tuple[List[ETFConfig], List[str]]:
    """사용자가 요청한 심볼만 추려내고 존재하지 않는 값은 따로 반환."""
    if not symbols:
        return list(ETF_LIST), []

    normalized = {symbol.strip().upper() for symbol in symbols if symbol.strip()}
    selected: List[ETFConfig] = []
    found_keys: set[str] = set()

    for config in ETF_LIST:
        yahoo_symbol = config.yahoo_symbol.upper()
        label = config.label.upper()
        if yahoo_symbol in normalized or label in normalized:
            selected.append(config)
            found_keys.update({yahoo_symbol, label})

    missing = sorted(normalized - found_keys)
    return selected, missing


def run_dca_backtest(
    config: ETFConfig,
    start: dt.date,
    end: dt.date,
    daily_contribution: float = DAILY_CONTRIBUTION,
) -> Dict[str, float]:
    """매 거래일 일정 금액을 납입했을 때의 누적 성과 계산."""
    prices = _load_history(config.yahoo_symbol, start, end)
    dividends = _load_dividend_series(config.yahoo_symbol, start, end)
    dividend_map = {_to_date(idx): float(val) for idx, val in dividends.items()}

    region_rules = TAX_RULES[config.region]
    buy_fee = region_rules["transaction_cost"]
    sell_fee = region_rules["transaction_cost"]
    dividend_tax = region_rules["dividend_tax"]

    units = 0.0
    net_invested = 0.0
    total_contribution = 0.0
    dividend_cash = 0.0
    trading_days = 0

    for ts, price in prices.items():
        ts_date = _to_date(ts)
        trading_days += 1
        total_contribution += daily_contribution
        investable = daily_contribution * (1 - buy_fee)
        units_bought = investable / price
        units += units_bought
        net_invested += investable

        dividend_per_share = dividend_map.get(ts_date, 0.0)
        if dividend_per_share:
            dividend_cash += units * dividend_per_share * (1 - dividend_tax)

    end_price = float(prices.iloc[-1])
    gross_value = units * end_price + dividend_cash
    final_value = gross_value * (1 - sell_fee)
    net_profit = final_value - total_contribution
    net_return_pct = (net_profit / total_contribution) * 100 if total_contribution else 0.0
    avg_cost = net_invested / units if units else np.nan

    return {
        "ETF": config.label,
        "거래일 수": trading_days,
        "총 납입액": total_contribution,
        "최종 평가액": final_value,
        "총 수익": net_profit,
        "총 수익률%": net_return_pct,
        "보유 수량": units,
        "평균 매입단가": avg_cost,
        "세후 배당금": dividend_cash,
        "최종 주가": end_price,
        "배당세율%": dividend_tax * 100,
        "거래비용%(편도)": buy_fee * 100,
    }


def run_backtests(
    start: dt.date,
    end: dt.date,
    daily_contribution: float,
    symbols: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, float]], List[str]]:
    """공통 로직을 묶어서 CLI/HTTP 에서 재사용."""
    configs, missing = _resolve_configs(symbols)
    rows: List[Dict[str, float]] = []
    errors: List[str] = []

    if missing:
        errors.append(f"알 수 없는 ETF: {', '.join(missing)}")

    if not configs:
        return rows, errors

    for config in configs:
        try:
            rows.append(run_dca_backtest(config, start, end, daily_contribution))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{config.label} ({config.yahoo_symbol}): {exc}")

    return rows, errors


@router.post("/etf-periodic-investment", response_model=BacktestResponse)
def etf_periodic_investment(request: BacktestRequest) -> BacktestResponse:
    """FastAPI 엔드포인트: Next.js 에서 POST 호출."""
    today = dt.date.today()
    start = request.start_date or (today - dt.timedelta(days=DEFAULT_LOOKBACK_DAYS))
    end = request.end_date or today

    if start > end:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    results, errors = run_backtests(start, end, request.daily_contribution, request.symbols)

    if not results and errors:
        raise HTTPException(status_code=404, detail=errors)

    return BacktestResponse(
        start_date=start,
        end_date=end,
        daily_contribution=request.daily_contribution,
        etf_count=len(results),
        results=results,
        errors=errors,
    )


def main() -> None:
    today = dt.date.today()
    start = today - dt.timedelta(days=DEFAULT_LOOKBACK_DAYS)
    end = today

    print("ETF 적립식(일 5,000원) 1년 백테스트")
    print(f"기간: {start.isoformat()} → {end.isoformat()}")
    print("각 ETF에 거래일마다 5,000원을 투자하고 추정 세금·수수료를 반영한 결과입니다.\n")

    rows, errors = run_backtests(start, end, DAILY_CONTRIBUTION)

    if rows:
        df = pd.DataFrame(rows)
        numeric_cols = [
            "총 납입액",
            "최종 평가액",
            "총 수익",
            "총 수익률%",
            "보유 수량",
            "평균 매입단가",
            "세후 배당금",
            "최종 주가",
            "배당세율%",
            "거래비용%(편도)",
        ]
        df[numeric_cols] = df[numeric_cols].applymap(lambda x: np.nan if pd.isna(x) else float(x))
        print(df.to_string(index=False, justify="center", float_format=lambda x: f"{x:,.2f}"))
    else:
        print("생성된 결과가 없습니다.")

    if errors:
        print("\n문제 발생:")
        for msg in errors:
            print(f"- {msg}")


if __name__ == "__main__":
    main()
