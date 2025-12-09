#!/usr/bin/env bash
set -euo pipefail

# --- 1: 从命令行参数获取端口号 ---
# ${1:? ...} 语法表示如果第1个参数为空，就报错退出
PORT=${1:?错误：请提供端口号作为第一个参数。用法: bash start.sh <端口号>}
# ------------------------------------

cd "$(dirname "$0")"

# --- 2: PID 文件名与端口号挂钩 ---
PIDFILE="lasvsim-${PORT}.pid"
# ------------------------------------

LOG_DIR="logs"
CMD="./opensim run --log_level WarnLevel --addr :${PORT} --saas_endpoint https://qianxing-api.risenlighten.com"

mkdir -p "${LOG_DIR}"

ts=$(date +%Y%m%d-%H%M%S)
# --- 3: 日志文件名也与端口号挂钩，避免冲突 ---
LOG_FILE="${LOG_DIR}/opensim-${PORT}-${ts}.log"
# ------------------------------------

echo "[start] 正在端口 ${PORT} 上启动 opensim..."
nohup bash -lc "${CMD}" >> "${LOG_FILE}" 2>&1 &

# 等待端口起来
echo "[start] 等待端口 ${PORT} 监听..."
for i in {1..15}; do
  sleep 1
  # --- 4: 所有lsof命令都使用变量 $PORT ---
  if lsof -t -i:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    NEW_PID=$(lsof -t -i:"${PORT}" -sTCP:LISTEN | head -n1)
    echo "${NEW_PID}" > "${PIDFILE}"
    echo "[start] opensim 在端口 ${PORT} 上启动成功, PID: ${NEW_PID}, 日志: ${LOG_FILE}"
    exit 0
  fi
  # ------------------------------------
done

echo "[start] 失败: 端口 ${PORT} 未能成功监听。请检查日志：${LOG_FILE}" >&2
exit 1