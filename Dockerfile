FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PYTHONUNBUFFERED=1
CMD ["python", "r100_live_deriv_demo_bot.py", "--mode", "high_confidence", "--stake", "1", "--max-trades", "3", "--warmup-candles", "120", "--execute-demo"]
