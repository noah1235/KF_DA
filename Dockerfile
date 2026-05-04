FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only dependency files
COPY pyproject.toml uv.lock* ./

# Copy only the stable C++ dependency
COPY src/kf_da/vp_floats ./src/kf_da/vp_floats

# Install deps + vpfloat, but not the main kf-da project
RUN uv sync --frozen --no-install-project || uv sync --no-install-project

# Now copy the frequently changing project code
COPY . .
