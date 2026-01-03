# ========== Stage 1: Development Environment (Target: dev) ==========
FROM ubuntu:24.04 AS dev

LABEL org.opencontainers.image.source="https://github.com/Li-Xiang-Ideal/SparseRREF/tree/application"
LABEL org.opencontainers.image.url="https://github.com/munuxi/SparseRREF"
LABEL org.opencontainers.image.documentation="https://github.com/munuxi/SparseRREF/blob/master/README.md"
LABEL org.opencontainers.image.description="Sparse Reduced Row Echelon Form (RREF) with row and column permutations in C++."

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libflint-dev \
    libtbb-dev \
    libmimalloc-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
CMD ["/bin/bash"]


# ========== Stage 2: Build Intermediate Layer (Target: builder) ==========
FROM dev AS builder

COPY . /app

RUN g++ main.cpp -o sparserref -O3 -std=c++20 -lflint -lgmp -ltbb -lmimalloc
RUN g++ mma_link.cpp -fPIC -shared -O3 -std=c++20 -o mathlink.so -I./mma -lflint -ltbb -lmimalloc

# ========== Stage 3: User Release Version (Target: release) ==========
FROM ubuntu:24.04 AS release

LABEL org.opencontainers.image.source="https://github.com/Li-Xiang-Ideal/SparseRREF/tree/application"
LABEL org.opencontainers.image.url="https://github.com/munuxi/SparseRREF"
LABEL org.opencontainers.image.documentation="https://github.com/munuxi/SparseRREF/blob/master/README.md"
LABEL org.opencontainers.image.description="Sparse Reduced Row Echelon Form (RREF) with row and column permutations in C++."

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libflint18t64 \
    libtbb12 \
    libmimalloc2.0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/sparserref /usr/local/bin/sparserref
COPY --from=builder /app/mathlink.so /usr/local/lib/mathlink.so

RUN ldconfig

CMD ["/bin/bash"]