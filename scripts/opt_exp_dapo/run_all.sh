#!/bin/bash

# 定义脚本列表
scripts=(
    /workspace/yiqiuguo/lsrl/scripts/opt_exp_dapo/base.sh
    /workspace/yiqiuguo/lsrl/scripts/opt_exp_dapo/base_lm_loss.sh
    /workspace/yiqiuguo/lsrl/scripts/opt_exp_dapo/base_kl_loss.sh
    /workspace/yiqiuguo/lsrl/scripts/opt_exp_dapo/base_adgrad_topk.sh
    /workspace/yiqiuguo/lsrl/scripts/opt_exp_dapo/base_adgrad_topp.sh
    /workspace/yiqiuguo/lsrl/scripts/opt_exp_dapo/base_adgrad_soft.sh
    /workspace/yiqiuguo/lsrl/scripts/opt_exp_dapo/base_gamma_05.sh
    /workspace/yiqiuguo/lsrl/scripts/opt_exp_dapo/base_decay_fw.sh
    /workspace/yiqiuguo/lsrl/scripts/opt_exp_dapo/base_adgrad_topk_decay_fw.sh
    /workspace/yiqiuguo/lsrl/scripts/opt_exp_dapo/fast_conn.sh
    /workspace/yiqiuguo/lsrl/scripts/opt_exp_dapo/on_policy_conn.sh
    /workspace/yiqiuguo/lsrl/scripts/opt_exp_dapo/m_on_policy_conn.sh
)

# 循环执行
for script in "${scripts[@]}"; do
    echo "----------------------------------------------------"
    echo "正在启动: $script"
    echo "----------------------------------------------------"
    
    # 执行脚本
    # 注意：这里直接运行，不管退出状态码是多少，循环都会继续
    bash "$script"
    
    # 打印执行状态（可选）
    if [ $? -ne 0 ]; then
        echo "警告: $script 执行过程中出错，正在跳向下一个..."
    else
        echo "成功: $script 执行完毕。"
    fi
done

echo "所有任务尝试运行结束。"