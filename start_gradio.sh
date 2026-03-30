#!/bin/bash
# Ultralytics YOLO Gradio 界面启动脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置 Python 路径
export PYTHONPATH="$SCRIPT_DIR/ultralytics/lib/python3.11/site-packages:$PYTHONPATH"

# 获取本机 IP 地址
IP_ADDR=$(hostname -I | awk '{print $1}')
[ -z "$IP_ADDR" ] && IP_ADDR="localhost"

# 清理可能残留的 7860 端口进程
kill $(lsof -t -i:7860) 2>/dev/null

# 启动 Gradio 界面
echo "🚀 正在启动 Ultralytics YOLO Gradio 界面..."
python3 "$SCRIPT_DIR/gradio_ui.py" &
PID=$!

# 捕获 Ctrl+C / kill 信号，退出时自动杀掉后台进程
cleanup() {
    echo ""
    echo "⏹️ 正在停止服务..."
    kill $PID 2>/dev/null
    wait $PID 2>/dev/null
    exit
}
trap cleanup INT TERM

# 轮询等待服务真正就绪（最多等 30 秒）
count=0
while ! curl -s --max-time 1 http://localhost:7860 >/dev/null 2>&1; do
    sleep 0.5
    count=$((count + 1))
    if [ $count -ge 60 ]; then
        echo "⚠️ 服务启动超时，请检查日志"
        break
    fi
done

if [ $count -lt 60 ]; then
    echo "✅ 服务已就绪，访问地址: http://${IP_ADDR}:7860"
fi

# 保持前台，等待 Python 进程结束
wait $PID
