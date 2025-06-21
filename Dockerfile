# Imhotep Neural Network Framework Dockerfile
# Multi-stage build for optimal size and security

# Build stage
FROM nvidia/cuda:12.2-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV RUST_VERSION=1.70.0
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    python3 \
    python3-pip \
    python3-dev \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain $RUST_VERSION
ENV PATH="/root/.cargo/bin:${PATH}"

# Install additional Rust components
RUN rustup component add rustfmt clippy

# Install Python dependencies for building
RUN pip3 install maturin[patchelf] numpy

# Set working directory
WORKDIR /app

# Copy source code
COPY Cargo.toml Cargo.lock ./
COPY pyproject.toml ./
COPY build.rs ./
COPY src/ ./src/
COPY csrc/ ./csrc/
COPY cuda/ ./cuda/
COPY python/ ./python/
COPY README.md LICENSE ./

# Build the Rust library with CUDA support
RUN cargo build --release --features max-performance,cuda

# Build Python bindings
RUN maturin build --release --features python,cuda

# Runtime stage
FROM nvidia/cuda:12.2-runtime-ubuntu22.04 AS runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libblas3 \
    liblapack3 \
    libopenblas0 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python runtime dependencies
RUN pip3 install --no-cache-dir \
    numpy>=1.20.0 \
    scipy>=1.7.0 \
    matplotlib>=3.4.0 \
    h5py>=3.1.0

# Create non-root user
RUN groupadd -r imhotep && useradd -r -g imhotep imhotep

# Create application directories
RUN mkdir -p /app /data /results && \
    chown -R imhotep:imhotep /app /data /results

# Copy built artifacts from builder stage
COPY --from=builder /app/target/release/imhotep-cli /usr/local/bin/
COPY --from=builder /app/target/wheels/*.whl /tmp/

# Install Python package
RUN pip3 install /tmp/*.whl && rm /tmp/*.whl

# Copy configuration and scripts
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/healthcheck.py /usr/local/bin/healthcheck.py
RUN chmod +x /usr/local/bin/entrypoint.sh

# Switch to non-root user
USER imhotep

# Set working directory
WORKDIR /app

# Expose port for any web interfaces
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 /usr/local/bin/healthcheck.py

# Entry point
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["imhotep-cli", "--help"]

# Development stage (optional)
FROM builder AS development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    valgrind \
    gdb \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python development dependencies
RUN pip3 install \
    pytest \
    pytest-benchmark \
    ipython \
    jupyter \
    black \
    isort \
    mypy

# Set up development environment
ENV RUST_LOG=debug
ENV RUST_BACKTRACE=full

# Keep the development container running
CMD ["tail", "-f", "/dev/null"]

# Minimal stage for CI/testing
FROM ubuntu:22.04 AS minimal

ENV DEBIAN_FRONTEND=noninteractive

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libblas3 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the CLI binary
COPY --from=builder /app/target/release/imhotep-cli /usr/local/bin/

# Test that the binary works
RUN imhotep-cli --version

CMD ["imhotep-cli"]

# Multi-architecture build information
LABEL org.opencontainers.image.title="Imhotep Neural Network Framework"
LABEL org.opencontainers.image.description="High-Performance Specialized Neural Network Framework"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/fullscreen-triangle/imhotep"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.vendor="Imhotep Development Team" 