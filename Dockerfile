FROM python:3.11-slim

# Disable Python bytecode generation and enable unbuffered output. Increase
# default pip timeouts and retries to make network operations more resilient.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set the working directory for the application code.
WORKDIR /app

# Use HTTPS mirrors and install OS dependencies in a single step. The original
# version of this file attempted to install packages within a retry loop but
# contained unreachable code after the loop's break statements. Consolidating
# the commands into a single `apt-get update` and `apt-get install` stage
# simplifies the Dockerfile and avoids hidden `break` conditions while still
# enabling retries via apt's built‑in options.
RUN set -eux; \
    # Replace HTTP sources with HTTPS to avoid network man-in-the-middle issues
    sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources; \
    # Update package lists and install ffmpeg and libgl1 (required for
    # torchvision/video transforms and OpenCV support) with retry options
    apt-get -o Acquire::Retries=5 \
            -o Acquire::http::Timeout=30 \
            -o Acquire::https::Timeout=30 \
            -o Acquire::ForceIPv4=true update; \
    apt-get install -y --no-install-recommends --fix-missing ffmpeg libgl1; \
    rm -rf /var/lib/apt/lists/*

# Copy project metadata and code into the container. Copying pyproject.toml and
# README.md before the source code leverages Docker layer caching: if only
# application code changes, package installation layers can remain cached.
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY configs /app/configs
COPY docs /app/docs

# Install Python dependencies. Torch and torchvision are installed from the
# official PyTorch CPU wheels to ensure compatibility. Installing in a single
# step avoids nested retry loops and simplifies failure handling.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu torchvision==0.20.1+cpu && \
    pip install --no-cache-dir -e .

# Provide a default command. `uav-vit --help` prints CLI usage and exits.
CMD ["uav-vit", "--help"]
