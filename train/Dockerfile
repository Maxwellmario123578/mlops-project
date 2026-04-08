FROM python:3.11-slim

# Utilisateur non-root
RUN groupadd -r traingroup && useradd -r -g traingroup -u 1001 trainuser
 
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
 
RUN mkdir -p /app/data /app/artifacts && \
    chown -R trainuser:traingroup /app
 
USER trainuser
CMD ["python", "preprocess.py"]
# train/requirements.txt
pandas==2.2.0
scikit-learn==1.4.0
boto3==1.34.0
numpy==1.26.0
