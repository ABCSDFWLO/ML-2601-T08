FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

RUN python -m pip install "pip<24.1" setuptools==65.5.0 wheel==0.38.4

WORKDIR /workspace

COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt
RUN pip install --no-cache-dir torch torchvision "pydantic<2"

CMD ["bash"]