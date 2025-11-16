from fastapi import FastAPI

from etfPeriodicInvestment import router as etf_router

app = FastAPI(title="Python API Gateway", version="1.0.0")
app.include_router(etf_router)


@app.get("/")
async def root():
    return {"message": "ETF Periodic Investment API is running"}
