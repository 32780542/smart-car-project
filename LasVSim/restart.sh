#!/usr/bin/env bash
set -euo pipefail

# ---  1: 从命令行参数获取端口号 ---
PORT=${1:?错误：请提供端口号作为第一个参数。用法: bash restart.sh <端口号>}
# ------------------------------------

cd "$(dirname "$0")"

# ---  2: PID 文件名与端口号挂钩 ---
PIDFILE="lasvsim-${PORT}.pid"
# ------------------------------------

echo "[restart] 正在检查并停止端口 ${PORT} 上的旧进程..."

# ---  3: lsof命令使用变量 $PORT ---
OLD_PIDS=$(lsof -t -i:"${PORT}" -sTCP:LISTEN || true)
# ------------------------------------

if [[ -z "${OLD_PIDS}" && -f "${PIDFILE}" ]]; then
  PID_IN_FILE=$(cat "${PIDFILE}" || true)
  if [[ -n "${PID_IN_FILE}" ]] && ps -p "${PID_IN_FILE}" >/dev/null 2>&1; then
    OLD_PIDS="${PID_IN_FILE}"
  fi
fi

if [[ -n "${OLD_PIDS}" ]]; then
  echo "[restart] 发现旧进程，正在发送 TERM 信号给: ${OLD_PIDS}"
  kill -TERM ${OLD_PIDS} || true
  for i in {1..10}; do
    sleep 1
    # ---  4: lsof命令使用变量 $PORT ---
    if ! lsof -t -i:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "[restart] 进程已成功停止。"
      break
    fi
    # ------------------------------------
    if [[ $i -eq 10 ]]; then
      echo "[restart] 进程未能在10秒内优雅停止，强制杀死: ${OLD_PIDS}"
      kill -KILL ${OLD_PIDS} || true
      sleep 1
    fi
  done
fi

[[ -f "${PIDFILE}" ]] && rm -f "${PIDFILE}"

echo "[restart] 正在端口 ${PORT} 上启动新进程..."
# ---  5: 调用 start.sh 时传递端口号 ---
sleep 10
bash ./start.sh "${PORT}"
# ------------------------------------