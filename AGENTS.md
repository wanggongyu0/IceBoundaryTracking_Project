# AGENTS.md

## 项目背景

本项目用于 1060 铝板表面电热膜加热除冰实验的红外图像处理。

目标：
1. 读取红外温度序列；
2. 对任意冰形，包括规则圆冰、不规则残冰、散点冰，追踪融冰前的冰边缘；
3. 在一定误差范围内判断完全融冰时间；
4. 一旦判断完全融冰，停止后续边界追踪，避免电热膜未覆盖区域造成温差误检。

## 当前主线代码

当前主线文件是：

ice_boundary_tracking_debug_v2.py

该文件是 v0.1 调试版，已经具备：
1. 读取 .mat/.npy/.npz/csv/txt 红外数据；
2. 支持 MATLAB v7.3 MAT 文件；
3. 统一数据维度为 [N,H,W]；
4. 支持 ROI 和 Ref ROI 选择；
5. 基于 Otsu 阈值生成 ice_mask；
6. 输出 overlay、result_table.csv、result_curve.png、summary.txt；
7. 基于面积比例估计完全融冰时间。

## 最高原则

1. 不允许重新生成整个项目。
2. 不允许删除已调试通过的函数。
3. 所有新功能都必须基于当前代码做增量修改。
4. 修改前必须先说明计划，等用户确认后再修改。
5. 优先新增函数，而不是重写旧函数。
6. 每次只完成一个小版本。
7. 修改后必须说明改了哪些文件、哪些函数、如何运行、如何验证、如何回退。

## 不允许随意修改的稳定函数

除非用户明确同意，不要修改以下函数：

- ensure_frames_nhw
- load_mat_v73_with_h5py
- load_thermal_data
- crop_roi
- clamp_roi
- select_rect_from_frame
- load_roi_config
- save_roi_config
- prepare_rois

## 可以逐步优化的函数

以下函数可以小步优化，但不能整体重写：

- segment_ice_frame
- morphology_clean
- remove_small_components
- save_overlay_image
- find_completion_time
- run_tracking

## 后续目标版本

v0.2：增加 config.yaml 外部配置  
v0.3：增加多连通域边界统计  
v0.4：适配散点冰和不规则冰  
v0.5：改进初始冰面积 A0 估计  
v0.6：多特征完全融冰判定  
v0.7：实时截断式边界追踪  
v0.8：可见光验证误差输出  
v0.9：多实验批处理  
v1.0：形成稳定可汇报算法版本