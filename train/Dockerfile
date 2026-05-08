FROM python:3.11-slim

RUN groupadd -r traingroup && useradd -r -g traingroup -u 1001 trainuser

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

RUN mkdir -p /app/data /app/artifacts && \
    chown -R trainuser:traingroup /app

USER trainuser
CMD ["python", "preprocess.py"]
