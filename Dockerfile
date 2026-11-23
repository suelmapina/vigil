# Scoring API image. Build after `make model`, which writes models/model.joblib.
#   docker build -t vigil-api .
#   docker run -p 8000:8000 vigil-api
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/vigil ./vigil
COPY models/model.joblib ./models/model.joblib
ENV PYTHONPATH=/app VIGIL_MODEL=/app/models/model.joblib

RUN useradd --create-home vigil
USER vigil
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen(\"http://localhost:8000/health\")"
CMD ["uvicorn", "vigil.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
