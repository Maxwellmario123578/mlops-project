FROM python:3.11-slim
 
# Sécurité : utilisateur non-root
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1000 appuser
 
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
 
# Dossier modèle accessible au user non-root
RUN mkdir -p /app/model_cache && chown -R appuser:appgroup /app
 
USER appuser
EXPOSE 8000
 
# Healthcheck intégré dans l'image
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
