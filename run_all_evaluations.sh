#!/bin/bash
# 这是一个自动生成的脚本，用于按顺序执行多个并行评估任务。
# 使用 'bash run_all_evaluations.sh' 来运行它。

set -e  # 如果任何命令失败，则立即退出脚本

# --- 任务 1: 评估场景 RAMEN_scene0000_D90_origin ---
python para_eval.py \
    --base_dir output/scannet/scene0000_00/RAMEN_scene0000_D90_origin \
    --config configs/scannet/scene0000.yaml \
    --gt_mesh output/scannet/scene0000_00/Centralized/agent_0/mesh_track5577_cull_occlusion.ply \
    --workers 40

# --- 任务 2: 评估场景 RAMEN_scene0000_D90_edge ---
python para_eval.py \
    --base_dir output/scannet/scene0000_00/RAMEN_scene0000_D90_edge \
    --config configs/scannet/scene0000.yaml \
    --gt_mesh output/scannet/scene0000_00/Centralized/agent_0/mesh_track5577_cull_occlusion.ply \
    --workers 40

# --- 任务 3: 评估场景 RAMEN_scene0000_D90_CP_edge ---
python para_eval.py \
    --base_dir output/scannet/scene0000_00/RAMEN_scene0000_D90_CP_edge \
    --config configs/scannet/scene0000.yaml \
    --gt_mesh output/scannet/scene0000_00/Centralized/agent_0/mesh_track5577_cull_occlusion.ply \
    --workers 40

# --- 任务 4: 评估场景 RAMEN_scene0000_D99_origin ---
python para_eval.py \
    --base_dir output/scannet/scene0000_00/RAMEN_scene0000_D99_origin \
    --config configs/scannet/scene0000.yaml \
    --gt_mesh output/scannet/scene0000_00/Centralized/agent_0/mesh_track5577_cull_occlusion.ply \
    --workers 40

# --- 任务 5: 评估场景 RAMEN_scene0000_D99_edge ---
python para_eval.py \
    --base_dir output/scannet/scene0000_00/RAMEN_scene0000_D99_edge \
    --config configs/scannet/scene0000.yaml \
    --gt_mesh output/scannet/scene0000_00/Centralized/agent_0/mesh_track5577_cull_occlusion.ply \
    --workers 40

# --- 任务 6: 评估场景 RAMEN_scene0000_D99_CP_edge ---
python para_eval.py \
    --base_dir output/scannet/scene0000_00/RAMEN_scene0000_D99_CP_edge \
    --config configs/scannet/scene0000.yaml \
    --gt_mesh output/scannet/scene0000_00/Centralized/agent_0/mesh_track5577_cull_occlusion.ply \
    --workers 40
