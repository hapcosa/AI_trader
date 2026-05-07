FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt /app/usdt_dominance_tv/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /app/usdt_dominance_tv/requirements.txt

COPY . /app/usdt_dominance_tv

CMD ["python", "-m", "usdt_dominance_tv.daemon"]
