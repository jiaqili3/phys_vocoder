#!/bin/bash
#SBATCH -J recoverpulse              # 作业名是 train
#SBATCH -p p-RTX2080      # 提交到 默认的defq 队列
#SBATCH -N 1                 # 使用2个节点
#SBATCH --ntasks-per-node=4  # 每个节点开启8个进程
#SBATCH --cpus-per-task=1    # 每个进程占用一个 cpu 核心
#SBATCH -t 50:00             # 任务最大运行时间是 50 分钟
python recover.py
