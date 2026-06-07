FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt /app/crypto_ohlcv_ccxt/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /app/crypto_ohlcv_ccxt/requirements.txt

COPY . /app/crypto_ohlcv_ccxt

CMD ["python", "-m", "crypto_ohlcv_ccxt.daemon"]
