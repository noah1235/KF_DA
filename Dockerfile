FROM python:3.14-slim

WORKDIR /app

RUN pip install uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Copy dependency files and pre-built vpfloat wheel
COPY pyproject.toml uv.lock* ./
COPY wheels/ ./wheels/

# Install deps (including pre-built vpfloat wheel) but not the main project
RUN uv sync --frozen --no-install-project

# Now copy the frequently changing project code
COPY . .


