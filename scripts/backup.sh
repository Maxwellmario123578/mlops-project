#!/bin/sh
set -e
DATE=$(date +%Y%m%d_%H%M)
BUCKET=$S3_BUCKET

echo "[$DATE] Demarrage backup..."

if [ -f "/backup/model_cache/model.pkl" ]; then
  aws s3 cp /backup/model_cache/model.pkl \
    s3://$BUCKET/backups/models/model_$DATE.pkl
  echo "[$DATE] Modele sauvegarde."
else
  echo "[$DATE] Aucun modele dans le cache - ignore."
fi

tar czf /tmp/prom_$DATE.tar.gz -C /backup/prometheus . 2>/dev/null || true
if [ -f /tmp/prom_$DATE.tar.gz ]; then
  aws s3 cp /tmp/prom_$DATE.tar.gz \
    s3://$BUCKET/backups/prometheus/prom_$DATE.tar.gz
  rm /tmp/prom_$DATE.tar.gz
  echo "[$DATE] Prometheus sauvegarde."
fi

echo "[$DATE] Backup termine avec succes."
