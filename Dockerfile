# GPU quote requires pynvml, which requires cuda, so use vllm image instead of python3
FROM vllm/vllm-openai@sha256:014a95f21c9edf6abe0aea6b07353f96baa4ec291c427bb1176dc7c93a85845c

ENV PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    SOURCE_DATE_EPOCH=0

# Install dependencies
WORKDIR /tmp

# Install packages via requirements.txt instead of poetry
# because of nv-ppcie-verifier requires some old version packages,
# which is not compatible with lots of current dependencies.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf requirements.txt

# Copy source code
WORKDIR /app
COPY src ./
COPY --chmod=664 .GIT_REV /etc/
EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
