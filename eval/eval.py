"""
【阅读指引】
====================================================================
文件位置: eval/eval.py
所属阶段: 评估（Evaluation）
上游依赖:
  - utils/datasets.py (dataset_pixel_depth_eval)  ← 数据集加载
  - model/modeling_qwen3_vl.py (Qwen3VLForConditionalGeneration) ← 模型推理
  - model/processing_qwen3_vl.py (AutoProcessor)  ← 多模态输入预处理

核心功能:
  加载训练好的 DepthVLM 模型，对测试集逐帧做推理，获取 pixel-level depth_pred，
  然后与 GT 深度图比较，计算 delta1 指标并写入 JSONL 结果文件。

内部结构:
  compute_delta1() ← 像素级 delta1 指标计算
  main()           ← 主流程：加载 → 分片 → 批量推理 → 指标计算 → 写结果

评估模式（eval_mode）:
  1) sparse-points：从 pred_map 中按 pixel_coords 采样若干点，与 GT 稀疏深度比较
  2) dense-depth-map：整张 pred_map resize 到 GT 分辨率后逐像素比较

多 GPU 加速:
  通过 --num_shards N --shard_id i 将数据集切分为 N 片，每个 shard 一个进程，
  最后用 merge_eval_results.py 合并结果。

建议阅读顺序:
  1. main() 的参数解析和模型加载部分
  2. 批量推理循环中 inputs 构造 → model forward → outputs.depth_pred
  3. delta1 计算的 sparse vs dense 两条分支
  4. 结果写出格式

关键术语:
  - delta1: 深度预测评估指标，max(pred/gt, gt/pred) < 1.25 的像素比例
  - sharding: 将大数据集切分给多个 GPU 并行评估
  - canonical_size: resize 后的统一分辨率，GT 原始分辨率可能不同
====================================================================
"""
import argparse
import logging
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from model import Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from utils.datasets import dataset_pixel_depth_eval


def compute_delta1(pred: np.ndarray, gt: np.ndarray) -> float:
    """【compute_delta1】计算像素级 delta1 评估指标

    原理: delta1 = 有效像素中 max(pred/gt, gt/pred) < 1.25 的比例
          值越大越好，1.0 表示完美预测；0.0 表示无任何有效像素
          阈值 1.25 是深度估计领域的标准惯例（也有用 1.05 / 1.10 / 1.25 / 1.56 多档）

    参数:
        pred (np.ndarray): 预测深度图，形状 (H, W) 或 1D 数组
        gt   (np.ndarray): GT 深度图，与 pred 形状相同

    返回:
        float: delta1 得分 (0.0 ~ 1.0)。当无有效像素时返回 0.0

    关键步骤:
        1. ravel() 展平为 1D，支持 dense-map 和 sparse-points 两种输入
        2. 筛选有效像素：gt > 0 且 pred > 0（depth=0 通常表示无效/遮挡区域）
        3. 计算双向比值 max(pred/gt, gt/pred)
        4. 统计比值 < 1.25 的比例
    """
    pred_flat = pred.ravel()
    gt_flat = gt.ravel()
    valid = (gt_flat > 0) & (pred_flat > 0)
    if valid.sum() == 0:
        return 0.0
    pred_v = pred_flat[valid]
    gt_v = gt_flat[valid]
    ratio = np.maximum(pred_v / gt_v, gt_v / pred_v)
    return float((ratio < 1.25).mean())


def main(args):
    """【main】评估脚本主函数

    整体流程:
      ① 加载 processor + DepthVLM 模型（device_map="auto" 自动分配 GPU/CPU）
      ② 加载评估数据集（dataset_pixel_depth_eval，支持 dense 和 sparse 两种模式）
      ③ 根据 num_shards / shard_id 切分数据，多 shard 并行加速
      ④ 批量推理：构造多模态 messages → processor 编码 → model forward → depth_pred
      ⑤ 逐样本计算 delta1：
         - sparse-points：按 pixel_coords 从 pred_map 取采样点，与 gt_depths 比较
         - dense-map：pred_map resize 到 GT 分辨率后逐像素比较
      ⑥ 写 JSONL 结果（单 shard 写 test_eval_results.jsonl，多 shard 写 test_eval_results_shard{i}.jsonl）
      ⑦ 打印最终平均 delta1

    参数:
        args (argparse.Namespace): 命令行参数，关键字段:
          - model_path (str)            : 模型目录（HuggingFace 格式）
          - image_folder (str)          : 图像根目录（分号分隔多路径）
          - json_path (str)             : 测试集 JSONL 路径（分号分隔多路径）
          - depth_root (str, optional)  : 原始深度图根目录
          - eval_mode (str)             : "auto" | "sparse" | "dense"
          - generate_text (bool)        : 是否同时生成文本回复
          - save_depth_maps (bool)      : 是否保存 pred/gt 为 .npy
          - max_save_depth_maps (int)   : 最多保存几张（-1 = 全保存）
          - bsz (int)                   : 推理 batch size
          - samples_to_eval (int)       : 评估样本数（0 = 全部）
          - num_shards / shard_id       : 多 GPU 分片参数
          - output_dir (str)            : 结果输出目录

    返回:
        无。结果写入 JSONL，最终 delta1 打印到控制台。
    """
    model_path = args.model_path
    image_folder = args.image_folder
    json_path = args.json_path

    if args.num_shards > 1:
        print(f"[Shard {args.shard_id}] Using CUDA devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")

    # ===== Load processor and model =====
    base_model = model_path
    processor = AutoProcessor.from_pretrained(base_model)
    processor.tokenizer.padding_side = "left"

    print("loading DepthLM with qwen3-vl architecture (Pixel Depth mode)")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.eval()

    # ===== Load dataset (supports semicolon-separated multi-jsonl) =====
    # json_path, image_folder, and depth_root all support semicolon-separated lists
    from utils.datasets import dataset_pixel_depth_eval
    dataset = dataset_pixel_depth_eval(
        json_path, image_folder,
        depth_root=args.depth_root,
        eval_mode=args.eval_mode,
    )
    print(f"dataset size = {len(dataset)}")

    samples_to_eval = len(dataset) if args.samples_to_eval == 0 else min(args.samples_to_eval, len(dataset))
    sampled_indices = list(range(samples_to_eval))

    # Sharding
    if args.num_shards > 1:
        total = len(sampled_indices)
        shard_size = (total + args.num_shards - 1) // args.num_shards
        start = args.shard_id * shard_size
        end = min(start + shard_size, total)
        sampled_indices = sampled_indices[start:end]
        print(f"[Shard {args.shard_id}/{args.num_shards}] Evaluating {len(sampled_indices)} samples")
    else:
        print(f"Evaluating {len(sampled_indices)} samples")

    # Output directory
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.json_path)
    os.makedirs(output_dir, exist_ok=True)
    if args.num_shards > 1:
        save_path = os.path.join(output_dir, f"test_eval_results_shard{args.shard_id}.jsonl")
    else:
        save_path = os.path.join(output_dir, "test_eval_results.jsonl")
    open(save_path, "w").close()

    # Directory for saved depth maps
    depth_maps_dir = None
    if args.save_depth_maps:
        depth_maps_dir = os.path.join(output_dir, "depth_maps")
        os.makedirs(depth_maps_dir, exist_ok=True)
        print(f"Depth maps will be saved to: {depth_maps_dir}")

    # ===== Metric accumulation =====
    all_delta1_scores = []
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)

    with torch.no_grad():
        for i in tqdm(range(0, len(sampled_indices), args.bsz)):
            batch_indices = sampled_indices[i : i + args.bsz]
            batch_messages = []
            for j in batch_indices:
                message = dataset[j]
                if message is not None:
                    batch_messages.append(message)
            if len(batch_messages) == 0:
                continue

            # ===== Build inputs =====
            messages_list = []
            for msg in batch_messages:
                messages = []
                if "system" in msg:
                    messages.append({
                        "role": "system",
                        "content": [{"type": "text", "text": msg["system"]}],
                    })
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": msg["image"]},
                        {"type": "text", "text": msg["problem"]},
                    ],
                })
                messages_list.append(messages)

            texts = [
                processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in messages_list
            ]
            image_inputs = []
            for m in messages_list:
                imgs, vids = process_vision_info(m)
                image_inputs.append(imgs)

            inputs = processor(
                text=texts,
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            if i == 0:
                for sample_idx, input_ids in enumerate(inputs.input_ids):
                    total_tokens = (input_ids != processor.tokenizer.pad_token_id).sum().item()
                    image_tokens = (input_ids == image_token_id).sum().item()
                    text_tokens = total_tokens - image_tokens
                    print(f"[Sample {sample_idx}] total_tokens={total_tokens}, "
                          f"image_tokens={image_tokens}, text_tokens={text_tokens}")

            # ===== Forward: obtain pixel-level depth =====
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
            )

            pixel_depth_preds = outputs.depth_pred  # list of (H, W) tensors
            
            if pixel_depth_preds is None:
                print(f"WARNING: No pixel depth predictions for batch {i}")
                continue

            if i == 0:
                print(f"pixel_depth_preds[0] shape: {pixel_depth_preds[0].shape}")

            # ===== Text generation (stage 1b models only) =====
            text_outputs = [""] * len(batch_messages)
            if args.generate_text:
                generated_ids = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    max_new_tokens=128,
                    do_sample=False,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                text_outputs = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

            # ===== Compute delta1 metric =====
            for b, msg in enumerate(batch_messages):
                pred_map = pixel_depth_preds[b].cpu().float().numpy()  # (H, W) or (1, H, W)
                while pred_map.ndim > 2:
                    pred_map = pred_map[0]  # remove extra leading dims to get (H, W)

                image_name = msg.get("image_name", "")
                pixel_coords = msg.get("pixel_coords", None)  # (N, 2) np.int64, each row [x, y]
                gt_depths = msg.get("gt_depths", None)          # (N,) np.float32
                gt_map = msg.get("pixel_depth_labels", None)    # (H, W) np.float32 or None

                # ===== Sparse-points mode: pixel_coords + gt_depths =====
                if pixel_coords is not None and gt_depths is not None:
                    pred_h, pred_w = pred_map.shape
                    gt_depths_jsonl_all = msg.get("gt_depths_jsonl", None)  # (N,) all original points
                    pred_at_points = []
                    gt_at_points = []
                    gt_jsonl_at_points = []  # jsonl GT aligned with gt_at_points
                    for k in range(len(pixel_coords)):
                        x, y = int(pixel_coords[k][0]), int(pixel_coords[k][1])
                        # Coordinates are 0-based; x is the column (width), y is the row (height)
                        if 0 <= x < pred_w and 0 <= y < pred_h:
                            pred_at_points.append(pred_map[y, x])
                            gt_at_points.append(gt_depths[k])
                            if gt_depths_jsonl_all is not None:
                                gt_jsonl_at_points.append(gt_depths_jsonl_all[k])
                        else:
                            # Out-of-bounds coordinate; skip this point
                            pass

                    pred_arr = np.array(pred_at_points, dtype=np.float32)
                    gt_arr = np.array(gt_at_points, dtype=np.float32)
                    gt_jsonl_arr = np.array(gt_jsonl_at_points, dtype=np.float32) if gt_jsonl_at_points else None
                    delta1 = compute_delta1(pred_arr, gt_arr)

                    if i == 0 and b == 0:
                        pixel_coords_orig = msg.get("pixel_coords_orig", None)
                        print(f"Sample 0 [sparse mode]: pred_map shape={pred_map.shape}, "
                              f"n_points={len(pixel_coords)}, valid_points={len(pred_at_points)}")
                        if pixel_coords_orig is not None and len(pixel_coords) > 0:
                            print(f"  coord transform example: orig={pixel_coords_orig[0].tolist()} -> "
                                  f"canonical={pixel_coords[0].tolist()}")
                        if len(pred_at_points) > 0:
                            print(f"  pred depths at points: min={pred_arr.min():.4f}, max={pred_arr.max():.4f}")
                            print(f"  gt depths (final, with fallback): min={gt_arr.min():.4f}, max={gt_arr.max():.4f}")
                            if gt_jsonl_arr is not None and len(gt_jsonl_arr) > 0:
                                print(f"  gt depths (from jsonl, original):  min={gt_jsonl_arr.min():.4f}, max={gt_jsonl_arr.max():.4f}")
                                # Count fallbacks: a point in gt_arr equal to gt_jsonl_arr is a fallback
                                n_fallback = int(np.sum(np.abs(gt_arr - gt_jsonl_arr) < 1e-6))
                                n_from_map = len(gt_arr) - n_fallback
                                print(f"  gt source: {n_from_map} from depth_map, {n_fallback} from jsonl (fallback)")
                                # Per-point comparison for the first few points
                                n_show = min(3, len(gt_arr))
                                for kk in range(n_show):
                                    src = "jsonl" if abs(gt_arr[kk] - gt_jsonl_arr[kk]) < 1e-6 else "depth_map"
                                    print(f"    point[{kk}]: gt={gt_arr[kk]:.4f} ({src}), gt_jsonl={gt_jsonl_arr[kk]:.4f}, "
                                          f"pred={pred_arr[kk]:.4f}")
                        print(f"  delta1={delta1:.4f}")
                        print(f"  text_output={text_outputs[b][:200]}")

                # ===== Full depth-map mode =====
                elif gt_map is not None:
                    # Diagnostic: check GT depth validity
                    gt_valid_count = (gt_map > 0).sum()
                    if gt_valid_count == 0:
                        print(f"WARNING [eval] sample index={batch_indices[b]}, "
                              f"image={image_name}: "
                              f"GT depth has NO valid pixels! "
                              f"shape={gt_map.shape}, min={gt_map.min():.4f}, max={gt_map.max():.4f}")

                    # If pred and gt differ in resolution, resize pred to gt resolution
                    if pred_map.shape != gt_map.shape:
                        pred_map_resized = cv2.resize(
                            pred_map, (gt_map.shape[1], gt_map.shape[0]),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    else:
                        pred_map_resized = pred_map

                    delta1 = compute_delta1(pred_map_resized, gt_map)

                    if i == 0 and b == 0:
                        pred_pos = pred_map[pred_map > 0]
                        pred_range_str = (f"[{pred_pos.min():.2f}, {pred_map.max():.2f}]m"
                                          if pred_pos.size > 0 else f"[NO_VALID_PIXELS, max={pred_map.max():.2f}]m")
                        print(f"Sample 0: pred_map shape={pred_map.shape}, range={pred_range_str}")
                        gt_pos = gt_map[gt_map > 0]
                        gt_range_str = (f"[{gt_pos.min():.2f}, {gt_map.max():.2f}]m"
                                        if gt_pos.size > 0 else f"[NO_VALID_PIXELS, max={gt_map.max():.2f}]m")
                        print(f"Sample 0: gt_map shape={gt_map.shape}, range={gt_range_str}")
                        print(f"Sample 0: delta1={delta1:.4f}")
                        print(f"Sample 0: text_output={text_outputs[b][:200]}")
                else:
                    delta1 = 0.0

                all_delta1_scores.append(delta1)

                # Save depth maps as .npy (controlled by max_save_depth_maps)
                depth_map_filename = ""
                if depth_maps_dir and image_name:
                    max_save = args.max_save_depth_maps
                    saved_count = getattr(args, '_saved_depth_count', 0)
                    if max_save < 0 or saved_count < max_save:
                        safe_name = image_name.replace("/", "_").replace(".jpg", "").replace(".png", "")
                        pred_npy_path = os.path.join(depth_maps_dir, f"{safe_name}_pred.npy")
                        np.save(pred_npy_path, pred_map.astype(np.float32))
                        depth_map_filename = f"{safe_name}_pred.npy"
                        if gt_map is not None:
                            gt_npy_path = os.path.join(depth_maps_dir, f"{safe_name}_gt.npy")
                            np.save(gt_npy_path, gt_map.astype(np.float32))
                        args._saved_depth_count = saved_count + 1

                with open(save_path, "a") as f:
                    record = {
                        "image_name": image_name,
                        "pred_shape": list(pred_map.shape),
                        "delta1": delta1,
                        "text_output": text_outputs[b],
                        "pred_min": round(float(pred_map[pred_map > 0].min()), 4) if (pred_map > 0).any() else 0,
                        "pred_max": round(float(pred_map.max()), 4),
                        "pred_mean": round(float(pred_map[pred_map > 0].mean()), 4) if (pred_map > 0).any() else 0,
                    }
                    if depth_map_filename:
                        record["depth_map_file"] = depth_map_filename
                    # Sparse-points mode additionally records per-point info
                    if pixel_coords is not None and gt_depths is not None:
                        record["n_eval_points"] = len(pixel_coords)
                        record["eval_mode"] = "sparse_points"
                    else:
                        record["eval_mode"] = "dense_map"
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ===== Final metric =====
    if all_delta1_scores:
        avg_delta1 = sum(all_delta1_scores) / len(all_delta1_scores)
        shard_info = f" [shard {args.shard_id}]" if args.num_shards > 1 else ""
        print(f"final delta_1{shard_info} = {avg_delta1:.6f}")
        print(f"Evaluating {len(all_delta1_scores)} samples")
    else:
        print("No samples were evaluated.")

    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="DepthLM Pixel-Level Depth evaluation.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--depth_root", type=str, default=None,
                        help="Root dir of original depth maps for GT loading")
    parser.add_argument("--generate_text", action="store_true",
                        help="Generate text output (use for stage 1b models trained with LM loss)")
    parser.add_argument("--eval_mode", type=str, default="auto",
                        choices=["auto", "sparse", "dense"],
                        help="Evaluation mode: 'auto' picks per-sample by jsonl fields (sparse-priority); "
                             "'sparse' forces sparse-points (pixel_coords + depth); "
                             "'dense' forces full depth-map (depth_path).")
    parser.add_argument("--save_depth_maps", action="store_true",
                        help="Save pred/gt depth maps as .npy files for visualization")
    parser.add_argument("--max_save_depth_maps", type=int, default=-1,
                        help="Max number of depth maps to save (-1 = all)")
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--samples_to_eval", type=int, default=0,
                        help="Number of samples to eval (0 = all)")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    main(args)
