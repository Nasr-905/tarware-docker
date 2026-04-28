FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    libgl1 \
    libglu1-mesa \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libfontconfig1 \
    fonts-dejavu-core \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install TA-RWARE from the local clone
COPY tarware/ /app/tarware/
RUN pip install --no-cache-dir /app/tarware/external/pyastar2d_TARWARE && \
    pip install --no-cache-dir pandas && \
    pip install --no-cache-dir /app/tarware

# Install server dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY server.py .

CMD Xvfb :99 -screen 0 2048x2048x24 -ac +extension GLX +render -noreset & \
    sleep 1 && \
    DISPLAY=:99 uvicorn server:app --host 0.0.0.0 --port 8000
