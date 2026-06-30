FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Claude Code CLI for the 'claude-code' subscription provider (ai_clients/
# claude_code_client.py shells out to `claude -p`). Distributed via npm, so we
# add Node. Gated by INSTALL_CLAUDE_CODE (default true) so a build can opt out
# to keep the image small when the subscription provider isn't used.
ARG INSTALL_CLAUDE_CODE=true
RUN if [ "$INSTALL_CLAUDE_CODE" = "true" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
        && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
        && apt-get install -y --no-install-recommends nodejs \
        && npm install -g @anthropic-ai/claude-code \
        && apt-get purge -y --auto-remove gnupg \
        && rm -rf /var/lib/apt/lists/* \
        && claude --version || true ; \
    fi

COPY requirements-web.txt ./requirements-web.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-web.txt

COPY . ./pineforge_ai

EXPOSE 8100

CMD ["uvicorn", "pineforge_ai.web.app:app", "--host", "0.0.0.0", "--port", "8100"]
