# Image chấm MBPP/MBPP+ với EvalPlus cố định phiên bản.
# python:3.11-slim có bản arm64 nên chạy native trên Mac chip Apple (không giả lập amd64 -> không bị timeout giả).
# run_benchmark tự build image này (tag agent0-evalplus:0.3.1) ở lần chấm đầu tiên với --evaluator docker.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "evalplus==0.3.1"
WORKDIR /w
