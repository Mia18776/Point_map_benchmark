#!/usr/bin/env python3
"""Write the metrics manual (an HTML <section>) for the full report.

Every metric column of a results.csv gets one row: meaning, direction, the
function that computes it, and where the definition comes from. The script
refuses to write if results.csv has a metric column the manual does not cover,
or the manual lists one that no longer exists.

    python ~/pmb_jobs/metrics_manual.py [results.csv] > metrics_manual.html
"""

import csv
import html
import sys

DEFAULT_CSV = "/home/woody/iwfa/iwfa137h/pmb_runs/1863900/results.csv"

# Where a definition comes from.
OFFICIAL = "官方代码"          # computed by map-anything's own functions
STANDARD = "通用指标，本框架实现"  # widely used in the literature; implemented here
OURS = "本框架设计"            # designed for this benchmark

UP, DOWN, NEUTRAL = "越大越好", "越小越好", "参考信息"

M = "metrics.py"
O = "official_metrics.py"

# (column, meaning, direction, function, origin, shown in the report?)
GROUPS = [
    ("位姿 pose/：相机的位置和朝向", [
        ("pose/auc_5", "每两张照片的相对位姿误差（旋转和平移方向取较大者）在 0–5° 范围内的累积分布面积，0–100。主要排名指标", UP, f"{O} · pose_auc", OFFICIAL, True),
        ("pose/auc_30", "同上，范围放宽到 0–30°。官方只报告 AUC@5", UP, f"{O} · pose_auc", OFFICIAL, False),
        ("pose/rot_err_deg_mean", "每两张照片之间相对旋转的误差（度），所有照片对的平均", DOWN, f"{M} · evaluate_poses", STANDARD, False),
        ("pose/rot_err_deg_median", "同上，取中位数，不受少数离谱的照片对影响", DOWN, f"{M} · evaluate_poses", STANDARD, False),
        ("pose/rra_5deg", "相对旋转误差小于 5° 的照片对比例（RRA@5）", UP, f"{M} · evaluate_poses", STANDARD, False),
        ("pose/trans_ang_err_deg_mean", "每两张照片之间平移方向的误差（度），平均。区分正反方向，比官方更严格", DOWN, f"{M} · evaluate_poses", STANDARD, False),
        ("pose/trans_ang_err_deg_median", "同上，取中位数", DOWN, f"{M} · evaluate_poses", STANDARD, False),
        ("pose/rta_5deg", "平移方向误差小于 5° 的照片对比例（RTA@5）", UP, f"{M} · evaluate_poses", STANDARD, False),
        ("pose/ate_rmse", "把预测的相机位置整体对齐到真值后，相机位置的均方根误差（米）", DOWN, f"{M} · evaluate_poses", STANDARD, False),
        ("pose/ate_rmse_rel", "同上，除以相机轨迹的大小，不同场景之间可以比较", DOWN, f"{M} · evaluate_poses", OURS, False),
    ]),
    ("内参 rays/：焦距和视场角", [
        ("rays/err_deg", "每个像素上预测的光线方向和真实光线方向的夹角（度），所有像素的平均", DOWN, f"{O} · ray_direction_error_deg", OFFICIAL, True),
    ]),
    ("官方定义 official/：和 MapAnything 论文直接对比", [
        ("official/pointmaps_abs_rel", "所有照片的 3D 点一起按平均距离归一化后，逐点的相对误差", DOWN, f"{O} · upstream_dense_n_view_metrics", OFFICIAL, False),
        ("official/pointmaps_inlier_thres_103", "同上，误差在 3% 以内的点所占比例", UP, f"{O} · upstream_dense_n_view_metrics", OFFICIAL, False),
        ("official/z_depth_abs_rel", "同一次归一化后，每个像素深度的相对误差。不单独对齐尺度", DOWN, f"{O} · upstream_dense_n_view_metrics", OFFICIAL, False),
        ("official/z_depth_inlier_thres_103", "同上，深度误差在 3% 以内的像素比例", UP, f"{O} · upstream_dense_n_view_metrics", OFFICIAL, False),
        ("official/pose_ate_rmse", "相机位置的均方根误差，归一化后的单位", DOWN, f"{O} · upstream_dense_n_view_metrics", OFFICIAL, False),
        ("official/metric_scale_abs_rel", "预测的场景大小和真实大小的相对误差。官方对所有模型都算，但只对输出真实尺度的模型有意义", DOWN, f"{O} · upstream_dense_n_view_metrics", OFFICIAL, False),
    ]),
    ("点云（相似变换对齐后）sim3/：整体 3D 形状", [
        ("sim3/accuracy_median_rel_extent", "精度：每个预测点到最近真值点的距离，中位数，除以场景大小。报告里的\"点云误差\"", DOWN, f"{M} · evaluate_pointmaps_sim3", STANDARD, True),
        ("sim3/accuracy_median", "同上，单位为米", DOWN, f"{M} · evaluate_pointmaps_sim3", STANDARD, False),
        ("sim3/accuracy_mean", "精度的平均值（米）", DOWN, f"{M} · evaluate_pointmaps_sim3", STANDARD, False),
        ("sim3/accuracy_mean_rel_extent", "精度的平均值，除以场景大小", DOWN, f"{M} · evaluate_pointmaps_sim3", STANDARD, False),
        ("sim3/completeness_median_rel_extent", "完整度：每个真值点到最近预测点的距离，中位数，除以场景大小。没有预测到的区域会被扣分", DOWN, f"{M} · evaluate_pointmaps_sim3", STANDARD, False),
        ("sim3/completeness_median", "同上，单位为米", DOWN, f"{M} · evaluate_pointmaps_sim3", STANDARD, False),
        ("sim3/completeness_mean", "完整度的平均值（米）", DOWN, f"{M} · evaluate_pointmaps_sim3", STANDARD, False),
        ("sim3/completeness_mean_rel_extent", "完整度的平均值，除以场景大小", DOWN, f"{M} · evaluate_pointmaps_sim3", STANDARD, False),
        ("sim3/chamfer_mean", "Chamfer 距离：精度和完整度平均值的平均（米）", DOWN, f"{M} · evaluate_pointmaps_sim3", STANDARD, False),
        ("sim3/chamfer_mean_rel_extent", "同上，除以场景大小", DOWN, f"{M} · evaluate_pointmaps_sim3", STANDARD, False),
        ("sim3/mae", "逐像素：预测点和同一像素的真值点之间的距离，平均（米）", DOWN, f"{M} · evaluate_pointmaps_sim3", STANDARD, False),
        ("sim3/median_ae", "同上，中位数（米）", DOWN, f"{M} · evaluate_pointmaps_sim3", STANDARD, False),
        ("sim3/mae_rel_extent", "逐像素距离的平均，除以场景大小", DOWN, f"{M} · evaluate_pointmaps_sim3", OURS, False),
        ("sim3/inlier_l2_0.02", "逐像素距离小于场景大小 2% 的像素比例", UP, f"{M} · evaluate_pointmaps_sim3", OURS, False),
        ("sim3/inlier_l2_0.05", "同上，阈值 5%", UP, f"{M} · evaluate_pointmaps_sim3", OURS, False),
        ("sim3/inlier_l2_0.1", "同上，阈值 10%", UP, f"{M} · evaluate_pointmaps_sim3", OURS, False),
        ("sim3/mae_lsq", "诊断用：改用普通最小二乘对齐（不剔除离群点）时的逐像素平均距离（米）", NEUTRAL, f"{M} · evaluate_pointmaps_sim3", OURS, False),
        ("sim3/median_ae_lsq", "诊断用：同上，中位数。和 sim3/median_ae 差很多说明有离群点", NEUTRAL, f"{M} · evaluate_pointmaps_sim3", OURS, False),
        ("sim3/mae_lsq_rel_extent", "诊断用：普通对齐的平均距离，除以场景大小", NEUTRAL, f"{M} · evaluate_pointmaps_sim3", OURS, False),
        ("sim3/median_ae_lsq_rel_extent", "诊断用：普通对齐的中位数距离，除以场景大小", NEUTRAL, f"{M} · evaluate_pointmaps_sim3", OURS, False),
        ("sim3/scale", "对齐时用的缩放系数（预测要放大多少倍才和真值一样大）", NEUTRAL, f"{M} · evaluate_pointmaps_sim3", OURS, False),
        ("sim3/num_points", "参与计算的像素数", NEUTRAL, f"{M} · evaluate_pointmaps_sim3", OURS, False),
    ]),
    ("点云（第一个相机坐标系）cam0/：形状加位姿", [
        ("cam0/rel_ae", "在第一个相机的坐标系下，两边各自按平均距离归一化后，逐点的相对误差。概念上接近 official/pointmaps_abs_rel", DOWN, f"{M} · evaluate_pointmaps_in_first_camera_frame", STANDARD, False),
        ("cam0/mae", "同一坐标系下逐点距离的平均（归一化单位）", DOWN, f"{M} · evaluate_pointmaps_in_first_camera_frame", STANDARD, False),
        ("cam0/median_ae", "同上，中位数", DOWN, f"{M} · evaluate_pointmaps_in_first_camera_frame", STANDARD, False),
        ("cam0/delta_1.03", "预测点和真值点到相机的距离之比在 1.03 以内的比例。接近 official/pointmaps_inlier_thres_103", UP, f"{M} · evaluate_pointmaps_in_first_camera_frame", STANDARD, False),
        ("cam0/delta_1.05", "同上，阈值 1.05", UP, f"{M} · evaluate_pointmaps_in_first_camera_frame", STANDARD, False),
        ("cam0/delta_1.25", "同上，阈值 1.25", UP, f"{M} · evaluate_pointmaps_in_first_camera_frame", STANDARD, False),
        ("cam0/inlier_l2_0.02", "逐点距离小于 0.02（归一化单位）的比例", UP, f"{M} · evaluate_pointmaps_in_first_camera_frame", OURS, False),
        ("cam0/inlier_l2_0.05", "同上，阈值 0.05", UP, f"{M} · evaluate_pointmaps_in_first_camera_frame", OURS, False),
        ("cam0/inlier_l2_0.1", "同上，阈值 0.1", UP, f"{M} · evaluate_pointmaps_in_first_camera_frame", OURS, False),
        ("cam0/num_points", "参与计算的像素数", NEUTRAL, f"{M} · evaluate_pointmaps_in_first_camera_frame", OURS, False),
    ]),
    ("深度 depth/：每个像素离相机多远（先按中位数对齐尺度）", [
        ("depth/abs_rel", "每个像素深度的相对误差，平均。报告里的\"深度误差\"。比官方的 z_depth_abs_rel 宽松", DOWN, f"{M} · evaluate_depth", STANDARD, True),
        ("depth/rmse", "深度的均方根误差（米）", DOWN, f"{M} · evaluate_depth", STANDARD, False),
        ("depth/delta_1.03", "预测深度和真实深度之比在 1.03 以内的像素比例", UP, f"{M} · evaluate_depth", STANDARD, False),
        ("depth/delta_1.05", "同上，阈值 1.05", UP, f"{M} · evaluate_depth", STANDARD, False),
        ("depth/delta_1.25", "同上，阈值 1.25（单目深度论文最常用的 δ<1.25）", UP, f"{M} · evaluate_depth", STANDARD, False),
        ("depth/scale", "对齐深度时用的缩放系数", NEUTRAL, f"{M} · evaluate_depth", OURS, False),
        ("depth/num_points", "参与计算的像素数", NEUTRAL, f"{M} · evaluate_depth", OURS, False),
    ]),
    ("真实尺度 metric/：只对输出米制尺度的模型", [
        ("metric/scale_abs_rel", "预测的场景大小和真实大小的相对误差。报告里的\"尺度误差\"", DOWN, f"{M} · evaluate_pointmaps_in_first_camera_frame", STANDARD, True),
        ("metric/scale_ratio", "预测大小除以真实大小。1 是完全正确，小于 1 是偏小", NEUTRAL, f"{M} · evaluate_pointmaps_in_first_camera_frame", OURS, False),
    ]),
    ("覆盖率 gt/、点云统计 cloud/", [
        ("gt/covered_ratio", "有真值的像素中，模型给出了有效点的比例。加遮罩的模型会低于 1", UP, f"{M} · evaluate_with_gt", OURS, False),
        ("gt/valid_ratio", "所有像素中，既有真值又有预测的比例", NEUTRAL, f"{M} · evaluate_with_gt", OURS, False),
        ("cloud/valid_ratio", "所有像素中，模型给出有效点的比例（不需要真值）", NEUTRAL, f"{M} · pointcloud_stats", OURS, False),
        ("cloud/num_points", "模型给出的有效点数", NEUTRAL, f"{M} · pointcloud_stats", OURS, False),
        ("cloud/extent_mean", "预测点云的大小：各点到中心的平均距离（模型自己的单位）", NEUTRAL, f"{M} · pointcloud_stats", OURS, False),
        ("cloud/bbox_diagonal", "预测点云外接盒的对角线长度（模型自己的单位）", NEUTRAL, f"{M} · pointcloud_stats", OURS, False),
    ]),
    ("多视图一致性 consistency/：不需要真值", [
        ("consistency/inlier_1.03", "把一张照片的点投影到另一张照片里，和那张照片自己预测的深度相比，比值在 1.03 以内的比例", UP, f"{M} · multiview_consistency", OURS, False),
        ("consistency/inlier_1.05", "同上，阈值 1.05", UP, f"{M} · multiview_consistency", OURS, False),
        ("consistency/inlier_1.25", "同上，阈值 1.25", UP, f"{M} · multiview_consistency", OURS, False),
        ("consistency/rel_depth_err_median", "投影后深度的相对差异，中位数", DOWN, f"{M} · multiview_consistency", OURS, False),
        ("consistency/rel_depth_err_mean", "同上，平均值。被遮挡的点也会计入，所以通常偏大", DOWN, f"{M} · multiview_consistency", OURS, False),
        ("consistency/overlap_ratio", "投影后落在另一张照片画面内的点的比例，反映照片之间的重叠程度", NEUTRAL, f"{M} · multiview_consistency", OURS, False),
        ("consistency/num_pairs", "参与计算的照片对数", NEUTRAL, f"{M} · multiview_consistency", OURS, False),
    ]),
    ("自检 sanity/：检查评测本身有没有算错，应接近 0", [
        ("sanity/pose_pointmap_rel_err", "用模型自己的位姿把相机坐标系的点变换到世界坐标系，和模型直接输出的世界坐标点比较。偏大说明位姿约定转换错了", DOWN, f"{M} · pose_pointmap_agreement", OURS, False),
        ("sanity/depth_pointmap_rel_err", "用模型自己的深度和内参重建相机坐标系的点，和模型直接输出的比较。偏大说明内参或深度约定错了", DOWN, f"{M} · pose_pointmap_agreement", OURS, False),
    ]),
    ("效率", [
        ("seconds_per_view", "处理每张照片的平均时间（秒），A100 上测得，不含加载模型", DOWN, "models.py · run_model", STANDARD, True),
        ("inference_seconds", "处理一整组照片的总时间（秒）", DOWN, "models.py · run_model", STANDARD, False),
        ("peak_memory_mb", "推理时 GPU 显存的峰值（MB）", DOWN, "models.py · run_model", STANDARD, True),
    ]),
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    with open(path, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    in_csv = {c for c in header if ("/" in c and not c.startswith("sampling/"))
              or c in ("seconds_per_view", "inference_seconds", "peak_memory_mb")}
    listed = [row[0] for _, rows in GROUPS for row in rows]
    missing, extra = in_csv - set(listed), set(listed) - in_csv
    if missing or extra or len(listed) != len(set(listed)):
        sys.exit(f"manual out of date - missing: {sorted(missing)}, gone: {sorted(extra)}")

    counts = {o: sum(1 for _, rows in GROUPS for r in rows if r[4] == o) for o in (OFFICIAL, STANDARD, OURS)}
    out = ['<section id="manual">',
           '  <h2><span class="n">11</span>指标手册</h2>',
           f'  <p>每组照片都会算出下面全部 {len(listed)} 个指标，保存在 <code>results.csv</code> 里。'
           '标 ★ 的是报告里使用的主要指标。</p>',
           '  <div class="tablewrap"><table>',
           '    <thead><tr><th>来源</th><th class="num">数量</th><th>说明</th></tr></thead><tbody>',
           f'      <tr><td>{OFFICIAL}</td><td class="num">{counts[OFFICIAL]}</td><td>直接调用 map-anything 的函数计算，和官方 benchmark 逐组一致</td></tr>',
           f'      <tr><td>{STANDARD}</td><td class="num">{counts[STANDARD]}</td><td>3D 重建、深度估计、位姿估计论文里常用的指标，计算代码是本框架写的</td></tr>',
           f'      <tr><td>{OURS}</td><td class="num">{counts[OURS]}</td><td>为这个评测专门设计：不需要真值的一致性、自检、覆盖率、诊断用的变体等</td></tr>',
           '    </tbody></table></div>',
           '  <p class="sub">"函数"一列指评测框架仓库里 <code>pointmap_bench/</code> 下的文件和函数，代码开头有更详细的说明。</p>']
    for title, rows in GROUPS:
        out += [f'  <h3>{html.escape(title)}</h3>', '  <div class="tablewrap"><table>',
                '    <thead><tr><th>指标</th><th>含义</th><th>方向</th><th>来源</th><th>函数</th></tr></thead><tbody>']
        for col, meaning, direction, func, origin, shown in rows:
            star = "★ " if shown else ""
            out.append(f'      <tr><td><code>{star}{html.escape(col)}</code></td><td>{html.escape(meaning)}</td>'
                       f'<td>{direction}</td><td>{origin}</td><td><code>{html.escape(func)}</code></td></tr>')
        out += ['    </tbody></table></div>']
    out.append('</section>')
    print("\n".join(out))


if __name__ == "__main__":
    main()
