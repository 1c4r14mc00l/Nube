FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

# Instala dependencias del sistema (incluyendo las gráficas)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements.txt e instala dependencias de Python
COPY scripts/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# El código se montará via volumen (configurado en compose)
#CMD ["python3", "/app/productor.py"]