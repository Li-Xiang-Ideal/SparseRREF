# ========== Stage 1: Build FLINT (Target: flint) ==========
FROM ubuntu:24.04 AS flint

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    m4 \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

ARG FLINT_CFLAGS="-O3 -march=x86-64-v3 -fPIC"

RUN wget https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz && \
    tar -xf gmp-6.3.0.tar.xz && \
    cd gmp-6.3.0 && \
    ./configure \
        --prefix=/usr/local \
        --enable-static \
        --enable-fat \
        --with-pic \
        --host=x86_64-pc-linux-gnu \
        CFLAGS="${FLINT_CFLAGS}" && \
    make -j$(nproc) && \
    make install

RUN wget https://www.mpfr.org/mpfr-4.2.2/mpfr-4.2.2.tar.xz && \
    tar -xf mpfr-4.2.2.tar.xz && \
    cd mpfr-4.2.2 && \
    ./configure \
        --prefix=/usr/local \
        --with-gmp=/usr/local \
        --enable-static \
        --with-pic \
        --host=x86_64-pc-linux-gnu \
        CFLAGS="${FLINT_CFLAGS}" && \
    make -j$(nproc) && \
    make install

RUN wget https://flintlib.org/download/flint-3.4.0.tar.gz && \
    tar -xf flint-3.4.0.tar.gz && \
    cd flint-3.4.0 && \
    ./configure \
        --prefix=/usr/local \
        --with-gmp=/usr/local \
        --with-mpfr=/usr/local \
        --enable-static \
        --enable-avx2 \
        --with-pic \
        --host=x86_64-pc-linux-gnu \
        CFLAGS="${FLINT_CFLAGS}" && \
    make -j$(nproc) && \
    make install

# ========== Stage 2: Extract Wolfram Headers (Target: wolfram) ==========
FROM wolframresearch/wolframengine:14.3 AS wolfram

USER root

RUN mkdir -p /export/wolfram

RUN cp /usr/local/Wolfram/WolframEngine/*/SystemFiles/IncludeFiles/C/*.h /export/wolfram

# ========== Stage 3: Development Environment (Target: dev) ==========
FROM ubuntu:24.04 AS dev

LABEL org.opencontainers.image.source="https://github.com/Li-Xiang-Ideal/SparseRREF/tree/application"
LABEL org.opencontainers.image.url="https://github.com/munuxi/SparseRREF"
LABEL org.opencontainers.image.documentation="https://github.com/munuxi/SparseRREF/blob/master/README.md"
LABEL org.opencontainers.image.description="Sparse Reduced Row Echelon Form (RREF) with row and column permutations in C++."

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libtbb-dev \
    libmimalloc-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=flint /usr/local/include /usr/local/include
COPY --from=flint /usr/local/lib /usr/local/lib
# COPY --from=wolfram /export/wolfram /usr/local/include/wolfram

RUN ldconfig

WORKDIR /app
CMD ["/bin/bash"]


# ========== Stage 4: Build Intermediate Layer (Target: builder) ==========
FROM dev AS builder

COPY . /app

COPY --from=wolfram /export/wolfram /usr/local/include/wolfram

RUN g++ main.cpp -o sparserref -O3 -std=c++20 -lflint -lgmp -ltbb -lmimalloc
RUN g++ sprreflink.cpp -fPIC -shared -O3 -std=c++20 -o sprreflink.so -I/usr/local/include/wolfram -Wl,-Bstatic -lflint -lmpfr -lgmp -Wl,-Bdynamic -ltbb -lmimalloc

WORKDIR /dist
RUN mkdir -p lib bin

RUN cp -P /usr/local/lib/libgmp.so* ./lib/ && \
    cp -P /usr/local/lib/libmpfr.so* ./lib/ && \
    cp -P /usr/local/lib/libflint.so* ./lib/ && \
    cp /app/sprreflink.so ./lib/ && \
    cp /app/sparserref ./bin/

RUN strip --strip-unneeded ./lib/* ./bin/*

# ========== Stage 5: User Release Version (Target: release) ==========
FROM ubuntu:24.04 AS release

LABEL org.opencontainers.image.source="https://github.com/Li-Xiang-Ideal/SparseRREF/tree/application"
LABEL org.opencontainers.image.url="https://github.com/munuxi/SparseRREF"
LABEL org.opencontainers.image.documentation="https://github.com/munuxi/SparseRREF/blob/master/README.md"
LABEL org.opencontainers.image.description="Sparse Reduced Row Echelon Form (RREF) with row and column permutations in C++."

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libtbb12 \
    libmimalloc2.0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=flint /usr/local/include /usr/local/include
COPY --from=builder /dist/lib /usr/local/lib
COPY --from=builder /dist/bin /usr/local/bin

RUN ldconfig

WORKDIR /app
CMD ["/bin/bash"]