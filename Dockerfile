# Explicit registry name avoids Podman's interactive registry prompt.
FROM docker.io/library/python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy HOME=/tmp \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libopenal1 libsdl2-2.0-0 libx11-6 libxext6 \
    libasound2 libsndfile1 libstdc++6 libgcc-s1 libglu1-mesa \
    ca-certificates && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu \
       'torch>=2.2,<3' && \
    python -m pip install -r requirements.txt
COPY pyproject.toml README.md ./
COPY corridor_qnn/ corridor_qnn/
COPY tests/ tests/
COPY pretrained/ pretrained/
COPY scripts/ scripts/
RUN python -m pip install --no-deps .
RUN mkdir -p outputs/models
ENTRYPOINT ["python","-m","corridor_qnn"]
CMD ["--help"]
