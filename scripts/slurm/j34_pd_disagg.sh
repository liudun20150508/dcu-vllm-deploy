#!/bin/bash
#SBATCH -J acc-j34-pddisagg
#SBATCH -p hx1hdnormal
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:1
#SBATCH -t 01:00:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j34-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j34-%j.err

# J34（优先 7）：真模型上的 **PD 分离** —— prefill 实例 / decode 实例，KV 走 NCCL 传
#   节点1 = kv_producer (prefill)，节点2 = kv_consumer (decode)
set -u
WS=/public/home/ac6ksiqdok/zksg-all/中科曙光
MODEL=$HOME/models/DeepSeek-V2-Lite
module purge
source /etc/profile.d/modules.sh
module load compiler/dtk/26.04
module load app/rccl/dtk-25.04/26.01.1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_NET_PLUGIN=/public/software/app/rccl/shca_rdma_plugins/v8/lib/librccl-net-shca.so
PY=/usr/local/python3.12/bin/python3
HOSTS=($(scontrol show hostname $SLURM_NODELIST))
N1=${HOSTS[0]}; N2=${HOSTS[1]}
PF_PORT=8100; DC_PORT=8200
echo "########## J34 PD 分离 on $N1(prefill) / $N2(decode) ($(date)) ##########"
echo "模型：$(du -sh $MODEL | cut -f1)"

ENV="source /etc/profile.d/modules.sh; module load compiler/dtk/26.04; module load app/rccl/dtk-25.04/26.01.1; export HSA_FORCE_FINE_GRAIN_PCIE=1; export NCCL_NET_PLUGIN=$NCCL_NET_PLUGIN"

VLLM_ARGS="--trust-remote-code --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
--max-model-len 4096 --enforce-eager --disable-log-requests"

# ---- prefill 实例（kv_producer）：算 KV 并主动 PUT ----
KVC_PF="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_rank\":0,\"kv_connector_extra_config\":{\"http_port\":\"${PF_PORT}\",\"send_type\":\"PUT_ASYNC\"}}"
echo "[起] prefill(producer) on $N1:$PF_PORT"
ssh -o StrictHostKeyChecking=no $N1 "$ENV; cd $WS/experiments && $PY -m vllm.entrypoints.openai.api_server --model $MODEL --served-model-name DeepSeek-V2-Lite --port $PF_PORT $VLLM_ARGS --kv-transfer-config '$KVC_PF'" > /tmp/j34_pf.log 2>&1 &
PF_PID=$!

# ---- decode 实例（kv_consumer）：接 KV 并解码 ----
KVC_DC="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_rank\":1,\"kv_connector_extra_config\":{\"http_port\":\"${DC_PORT}\",\"send_type\":\"PUT_ASYNC\"}}"
echo "[起] decode(consumer) on $N2:$DC_PORT"
ssh -o StrictHostKeyChecking=no $N2 "$ENV; cd $WS/experiments && $PY -m vllm.entrypoints.openai.api_server --model $MODEL --served-model-name DeepSeek-V2-Lite --port $DC_PORT $VLLM_ARGS --kv-transfer-config '$KVC_DC'" > /tmp/j34_dc.log 2>&1 &
DC_PID=$!

echo
echo "=== 等两个实例 ready（最多 10 分钟）==="
ready_pf=0; ready_dc=0
for i in $(seq 1 60); do
  sleep 10
  if [ "$ready_pf" = 0 ]; then
    curl -s --max-time 3 "http://${N1}:${PF_PORT}/health" >/dev/null 2>&1 && ready_pf=1 && echo "  prefill ready（第 $((i*10)) 秒）"
  fi
  if [ "$ready_dc" = 0 ]; then
    curl -s --max-time 3 "http://${N2}:${DC_PORT}/health" >/dev/null 2>&1 && ready_dc=1 && echo "  decode  ready（第 $((i*10)) 秒）"
  fi
  [ "$ready_pf" = 1 ] && [ "$ready_dc" = 1 ] && break
done

if [ "$ready_pf" = 0 ] || [ "$ready_dc" = 0 ]; then
  echo "  ✗ 实例没起来（prefill=$ready_pf decode=$ready_dc）—— 看日志"
  echo "--- prefill 日志尾部 ---"; tail -20 /tmp/j34_pf.log
  echo "--- decode 日志尾部 ---"; tail -20 /tmp/j34_dc.log
  kill $PF_PID $DC_PID 2>/dev/null
  exit 0
fi

echo
echo "=== 跑两段式客户端：prefill → 传 KV → decode ==="
PREFILL_URL="http://${N1}:${PF_PORT}" DECODE_URL="http://${N2}:${DC_PORT}" \
VLLM_MODEL=DeepSeek-V2-Lite \
  $PY $WS/experiments/pd_disagg_client.py 2>&1 | tail -30

echo
echo "=== 两边的 connector 日志（看 KV 有没有真的传）==="
echo "--- prefill(producer) ---"; grep -a -E "P2P|PUT|send|KV" /tmp/j34_pf.log | tail -8
echo "--- decode(consumer)  ---"; grep -a -E "P2P|GET|recv|KV|load" /tmp/j34_dc.log | tail -8

kill $PF_PID $DC_PID 2>/dev/null
echo "########## J34 done $(date) ##########"
