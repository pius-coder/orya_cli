FROM python:3.12-slim AS base
WORKDIR /app/agent
COPY agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY agent/ ./agent/
COPY db/init.sql ./db/
EXPOSE 5001
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/health')" || exit 1
CMD ["uvicorn", "agent.main:app", "--host", "0.0.0.0", "--port", "5001"]
