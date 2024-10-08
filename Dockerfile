# The builder image, used to build the virtual environment
FROM python:3.10-slim-buster as builder

RUN apt-get update && apt-get install -y git

RUN pip install poetry==1.4.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

ENV HOST=0.0.0.0
ENV LISTEN_PORT 8000
EXPOSE 8000

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

# Install requirements from demo_app/requirements.txt
COPY demo_app/requirements.txt ./demo_app/
RUN pip install -r demo_app/requirements.txt && rm -rf /root/.cache/pip/* 

# The runtime image, used to just run the code provided its virtual environment
FROM python:3.10-slim-buster as runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY ./demo_app ./demo_app
COPY ./.chainlit ./.chainlit
COPY chainlit.md ./

CMD ["chainlit", "run", "demo_app/main.py"]
