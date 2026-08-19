# syntax=docker/dockerfile:1.4
# openfilter-base = python:3.14-slim + all outstanding Debian security patches
# (rebuilt weekly): provides the PYTHONDONTWRITEBYTECODE/PYTHONUNBUFFERED env, the
# appuser account, and /app (WORKDIR) + /app/logs — so none of that is repeated here.
FROM plainsightai/openfilter-base:py3.14

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

USER appuser
CMD ["python", "-m", "filter_frame_dedup.filter"]
