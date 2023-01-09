FROM python:3.8.0-slim-buster

RUN apt update \
    && apt install -y --no-install-recommends \
    curl \
    ca-certificates \
    bash-completion \
    libgomp1 \
    g++ \
    gcc \
    git \
    make \
    graphviz \
    libopenblas-dev \
    python3-tk \
    && apt-get autoremove -y \
    && apt-get autoclean \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV SHELL /bin/bash

ENV POETRY_VERSION="1.1.13"

ENV POETRY_CACHE /work/.cache/poetry

ENV POETRY_HOME /usr/local/

ENV POETRY_VIRTUALENVS_PATH=$POETRY_CACHE

RUN curl -sSL https://install.python-poetry.org | python -


ENV PIP_CACHE_DIR /work/.cache/pip

ENV JUPYTER_RUNTIME_DIR /work/.cache/jupyter/runtime

ENV JUPYTER_CONFIG_DIR /work/.cache/jupyter/config

RUN git config --global --add safe.directory '*'

CMD ["bash", "-l"]