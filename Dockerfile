FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

COPY requirements-web.txt ./requirements-web.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-web.txt

COPY . ./pineforge_ai

EXPOSE 8100

CMD ["uvicorn", "pineforge_ai.web.app:app", "--host", "0.0.0.0", "--port", "8100"]
