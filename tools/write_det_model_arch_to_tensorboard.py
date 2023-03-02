import argparse
import copy
import os

import mmcv
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
# from torchpack import distributed as dist
from torchpack.utils.config import configs
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--out-dir", type=str, default="runs")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    # build the model
    model = build_model(cfg.model)
    model.cuda()
    model.eval()

    # make a input
    # build the dataloader
    dataset = build_dataset(cfg.data["val"])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        num_gpus=1,
        dist=False,
        shuffle=False,
    )

    for i, data in enumerate(dataflow):
        print(i)
        if i == 20:
            break

    data_list = [
        data["img"].data[0].cuda(),
        data["points"].data[0][0].cuda().unsqueeze(0),
        data["camera2ego"].data[0].cuda(),
        data["lidar2ego"].data[0].cuda(),
        data["lidar2camera"].data[0].cuda(),
        data["lidar2image"].data[0].cuda(),
        data["camera_intrinsics"].data[0].cuda(),
        data["camera2lidar"].data[0].cuda(),
        data["img_aug_matrix"].data[0].cuda(),
        data["lidar_aug_matrix"].data[0].cuda(),
        torch.randn([1, 6, 3, 4], dtype=torch.float32).cuda(),
    ]

    # write to tensorboard
    writer = SummaryWriter(args.out_dir)
    writer.add_graph(model, input_to_model=data_list)
    writer.close()



if __name__ == "__main__":
    main()
