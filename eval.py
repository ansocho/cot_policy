"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# try to import robosuite task zoo to include those envs in the robosuite registry
try:
    import robosuite_task_zoo
    import mimicgen
except ImportError:
    pass


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-gs", "--global_seed", required=True, type=int)
@click.option("-es", "--env_seed", required=True, type=int)
@click.option("-d", "--device", default="cuda:1")
@click.option("-is", "--num_inference_steps", default=2)
def main(
    checkpoint="/media/storage/soho/exps/outputs/2024.10.04/19.28.36_train_fm_unet_image_pusht_image_None/checkpoints/epoch=0860-test_mean_score=0.928.ckpt",
    output_dir="exps/outputs/pusht_eval_output",
    global_seed=1000,
    env_seed=100000,
    device="cuda:1",
    num_inference_steps=2,
):

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cfg.training.seed = global_seed
    cfg.training.device = device
    # cfg.outputs_dir = output_dir = "/media/storage/soho/exps/outputs/"
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy = policy.to(device)
    policy.eval()

    cfg.task.env_runner.test_start_seed = env_seed
    print(f"Policy initially had {policy.num_inference_steps} inference steps")
    policy.num_inference_steps = num_inference_steps

    # run eval
    cfg.task.env_runner.n_test = 50
    cfg.task.env_runner.n_train = 0
    cfg.task.env_runner.n_envs = None
    cfg.task.env_runner.n_train_vis = 0
    cfg.task.env_runner.n_test_vis = 0

    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
    runner_log = env_runner.run(policy)

    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, "eval_log.json")
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)


if __name__ == "__main__":
    main()