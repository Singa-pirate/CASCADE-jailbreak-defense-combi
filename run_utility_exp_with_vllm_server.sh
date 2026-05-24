#!/bin/bash
#SBATCH --job-name=CASCADE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=72:00:00
#SBATCH --mem=32G

export OPENAI_CLIENT_CONFIG_PATH="common/local_configs.yaml"
source venv/bin/activate

# Start vLLM in the background
srun vllm serve hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype auto \
  --api-key token-abc123 \
  > slurm-vllm.log 2>&1 &

echo "Waiting for vLLM server to be ready..."
for i in {1..60}; do
  if curl -s "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
    echo "vLLM server is up."
    break
  fi
  sleep 10
done

python main.py config5.yaml
