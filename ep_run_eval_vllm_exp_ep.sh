#!/bin/bash

# 1. 判断并带 -u 参数创建 tmux 窗口/会话
if [ -z "$TMUX" ]; then
    tmux -u new-session -d -s ep_session -n ep
else
    tmux -u new-window -d -n ep
fi

# 2. 发送环境初始化指令
tmux send-keys -t ep "source /root/conda_envs/vllm/bin/activate" C-m
tmux send-keys -t ep "cd /workspace/yiqiuguo/lsrl/" C-m

# 3. 核心修改：带上日志重定向
# "2>&1" 表示把错误信息也一并捕获
# "tee run_ep.log" 表示把输出同时打印在 tmux 屏幕上，并写入 run_ep.log 文件
tmux send-keys -t ep "bash /workspace/yiqiuguo/lsrl/run_eval_vllm_exp_ep.sh --RUN_SEED 42 --NODE_TOTAL 7 --NODE_RANK 6 2>&1 | tee run_ep.log" C-m

echo "任务已在 tmux 后台启动，并正在写入 run_ep.log"

# ==========================================
# 4. 视角控制（根据你的需求选择保留哪一段）
# ==========================================

# 【选项 A：直接切进 tmux 窗口看进度】（你之前的做法）
if [ -z "$TMUX" ]; then
    tmux -u attach-session -t ep_session
else
    tmux select-window -t ep
fi

# 【选项 B：如果不切进 tmux，想直接在原本的 bash 里实时看日志】
# 如果你想用这个，就把上面【选项 A】的 if 判断全部注释掉，换成下面这句：
# tail -f /workspace/yiqiuguo/lsrl/run_ep.log