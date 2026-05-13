#!/bin/bash


# 启动第一个模型：Qwen3-VL-8B-Instruct
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
  --model /mnt/Qwen3-VL-8B-Instruct/ \
  --host 0.0.0.0 \
  --port 8888 \
  --served-model-name Qwen3-VL-8B-Instruct \
  --gpu-memory-utilization 0.52 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --trust-remote-code \
  --enable-prefix-caching \
  --enforce-eager \
  --disable-log-requests &  # 减少日志噪音

PID1=$!
echo "Started Qwen3-VL-8B-Instruct on port 8888 with PID: $PID1"

# 等待第一个模型加载并稳定
echo "Waiting for first model to load..."
sleep 45

# 检查第一个模型是否成功加载
if ! kill -0 $PID1 2>/dev/null; then
    echo "ERROR: First model failed to start"
    exit 1
fi

echo "First model loaded successfully, starting second model..."

# 监控脚本
cat > monitor_gpu.sh << 'EOF'
#!/bin/bash
while true; do
    echo "=== $(date) ==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    sleep 10
done
EOF

chmod +x monitor_gpu.sh

echo "Both models started. Monitor with: ./monitor_gpu.sh"
echo "To stop: kill $PID1"

# 等待两个进程
wait $PID1