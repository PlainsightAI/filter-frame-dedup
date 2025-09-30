# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd -ms /bin/bash appuser
WORKDIR /app

# Install pip + filter-frame-dedup at version from VERSION file
RUN --mount=type=bind,source=VERSION,target=/tmp/VERSION,ro \
    set -eux; \
    RAW="$(head -n1 /tmp/VERSION)"; \
    # strip optional leading v/V and whitespace
    PKG_VERSION="$(printf '%s' "$RAW" | tr -d ' \t\r\n' | sed 's/^[vV]//')"; \
    [ -n "$PKG_VERSION" ] || { echo "VERSION file is empty"; exit 1; }; \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      --index-url https://python.openfilter.io/simple \
      --extra-index-url https://pypi.org/simple \
      "filter-frame-dedup==${PKG_VERSION}"

# Create a writable logs dir and hand over /app to appuser
RUN mkdir -p /app/logs && chown -R appuser:appuser /app

USER appuser
CMD ["python", "-m", "filter_frame_dedup.filter"]
