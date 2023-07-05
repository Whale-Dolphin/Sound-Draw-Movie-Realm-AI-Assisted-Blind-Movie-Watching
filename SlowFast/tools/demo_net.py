#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import time
import torch
import tqdm

from slowfast.utils import logging
from slowfast.visualization.async_predictor import AsyncDemo, AsyncVis
from slowfast.visualization.ava_demo_precomputed_boxes import (
    AVAVisualizerWithPrecomputedBox,
)
from slowfast.visualization.demo_loader import ThreadVideoManager, VideoManager
from slowfast.visualization.predictor import ActionPredictor
from slowfast.visualization.video_visualizer import VideoVisualizer

import pandas as pd

logger = logging.get_logger(__name__)


def run_demo(cfg, frame_provider):
    """
    Run demo visualization.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        frame_provider (iterator): Python iterator that return task objects that are filled
            with necessary information such as `frames`, `id` and `num_buffer_frames` for the
            prediction and visualization pipeline.
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    # Print config.
    logger.info("Run demo with config:")
    logger.info(cfg)
    common_classes = (
        cfg.DEMO.COMMON_CLASS_NAMES
        if len(cfg.DEMO.LABEL_FILE_PATH) != 0
        else None
    )

    video_vis = VideoVisualizer(
        num_classes=cfg.MODEL.NUM_CLASSES,
        class_names_path=cfg.DEMO.LABEL_FILE_PATH,
        top_k=cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS,
        thres=cfg.DEMO.COMMON_CLASS_THRES,
        lower_thres=cfg.DEMO.UNCOMMON_CLASS_THRES,
        common_class_names=common_classes,
        colormap=cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
        mode=cfg.DEMO.VIS_MODE,
    )

    async_vis = AsyncVis(video_vis, n_workers=cfg.DEMO.NUM_VIS_INSTANCES)

    if cfg.NUM_GPUS <= 1:
        model = ActionPredictor(cfg=cfg, async_vis=async_vis)
    else:
        model = AsyncDemo(cfg=cfg, async_vis=async_vis)

    seq_len = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE

    assert (
        cfg.DEMO.BUFFER_SIZE <= seq_len // 2
    ), "Buffer size cannot be greater than half of sequence length."
    num_task = 0

    data_list = []

    # Start reading frames.
    frame_provider.start()
    for able_to_read, task in frame_provider:
        if not able_to_read:
            break
        if task is None:
            time.sleep(0.02)
            continue
        num_task += 1

        model.put(task)
        
        try:
            task = model.get()
            num_task -= 1
            yield task

            data_action_preds = []
            data_bboxes = []

            for action_pred in task.action_preds:
                data_action_preds.append(action_pred)
            data_action_preds_max = []
            for i in range(len(data_action_preds)):
                max = 0
                max_index = 0
                max_1 = 0
                max_index_1 = 0
                max_2 = 0
                max_index_2 = 0
                for j in range(len(data_action_preds[i])):
                    
                    if data_action_preds[i][j] > max:
                        max_1 = max
                        max = data_action_preds[i][j]
                        max_index_1 = max_index
                        max_index = j
                    elif data_action_preds[i][j] > max_1:
                        max_2 = max_1
                        max_index_2 = max_index_1
                        max_1 = data_action_preds[i][j]
                        max_index_1 = j
                    elif data_action_preds[i][j] > max_2:
                        max_2 = data_action_preds[i][j]
                        max_index_2 = j
                data_max = []
                if max_2 > 0.75:
                    data_max = ([(max, max_index),(max_1, max_index_1),(max_2, max_index_2)])
                elif max_1 > 0.75:
                    data_max = ([(max, max_index),(max_1, max_index_1)])
                else:
                    data_max = ([(max, max_index)])
                data_action_preds_max.append(data_max)
            for bboxes in task.bboxes:
                data_bboxes.append(bboxes)
            for i in range(len(data_action_preds_max)):
                data_list.append({
                    'id': task.id,
                    'num_buffer_frames': task.num_buffer_frames,
                    'bboxes': data_bboxes[i],
                    'action_pred': data_action_preds_max[i],
                })
            df = pd.DataFrame(data_list)
            df.to_csv('C:/Users/17280/.conda/slowfast/output/output.csv', mode='a', header=False, index=False)
            data_list = []

        except IndexError:

            data_action_preds = []
            data_bboxes = []

            for action_pred in task.action_preds:
                data_action_preds.append(action_pred)
            data_action_preds_max = []
            for i in range(len(data_action_preds)):
                max = 0
                max_index = 0
                max_1 = 0
                max_index_1 = 0
                max_2 = 0
                max_index_2 = 0
                for j in range(len(data_action_preds[i])):
                    
                    if data_action_preds[i][j] > max:
                        max_1 = max
                        max = data_action_preds[i][j]
                        max_index_1 = max_index
                        max_index = j
                    elif data_action_preds[i][j] > max_1:
                        max_2 = max_1
                        max_index_2 = max_index_1
                        max_1 = data_action_preds[i][j]
                        max_index_1 = j
                    elif data_action_preds[i][j] > max_2:
                        max_2 = data_action_preds[i][j]
                        max_index_2 = j
                data_max = []
                if max_2 > 0.75:
                    data_max = ([(max, max_index),(max_1, max_index_1),(max_2, max_index_2)])
                elif max_1 > 0.75:
                    data_max = ([(max, max_index),(max_1, max_index_1)])
                else:
                    data_max = ([(max, max_index)])
                data_action_preds_max.append(data_max)
            for bboxes in task.bboxes:
                data_bboxes.append(bboxes)
            for i in range(len(data_action_preds_max)):
                data_list.append({
                    'id': task.id,
                    'num_buffer_frames': task.num_buffer_frames,
                    'bboxes': data_bboxes[i],
                    'action_pred': data_action_preds_max[i],
                })
            df = pd.DataFrame(data_list)
            df.to_csv('C:/Users/17280/.conda/slowfast/output/output.csv', mode='a', header=False, index=False)
            data_list = []
            
            continue

    while num_task != 0:
        try:
            task = model.get()
            num_task -= 1
            yield task
        except IndexError:
            continue


def demo(cfg):
    """
    Run inference on an input video or stream from webcam.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # AVA format-specific visualization with precomputed boxes.
    if cfg.DETECTION.ENABLE and cfg.DEMO.PREDS_BOXES != "":
        precomputed_box_vis = AVAVisualizerWithPrecomputedBox(cfg)
        precomputed_box_vis()
    else:
        start = time.time()
        if cfg.DEMO.THREAD_ENABLE:
            frame_provider = ThreadVideoManager(cfg)
        else:
            frame_provider = VideoManager(cfg)

        for task in tqdm.tqdm(run_demo(cfg, frame_provider)):
            frame_provider.display(task)

        frame_provider.join()
        frame_provider.clean()
        logger.info("Finish demo in: {}".format(time.time() - start))
