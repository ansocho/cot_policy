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
import omegaconf as OmegaConf
import numpy as np
from cot_policy.common.tv_utils import calculate_dtw_variance_from_dba

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
@click.option("-d", "--device", default="cuda:0")
@click.option("-is", "--num_inference_steps", default=2)
def main(
    checkpoint="/media/storage/soho/exps/outputs/2024.10.04/19.28.36_train_fm_unet_image_pusht_image_None/checkpoints/epoch=0860-test_mean_score=0.928.ckpt",
    output_dir="exps/outputs/pusht_eval_output",
    global_seed=1000,
    env_seed=100000,
    device="cuda:0",
    num_inference_steps=2,
):
    # if os.path.exists(output_dir):
    #     click.confirm(
    #         f"Output path {output_dir} already exists! Overwrite?", abort=True
    #     )
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

    initial_pos_idx = 0
    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
    if "maze-2d" in cfg.task.task_name:
        log_prefix = (
            cfg.policy_type + "_" + str(num_inference_steps) + "_" + cfg.task.task_name
        )

        runner_log = env_runner.run(
            policy,
            initial_pos=initial_pos_idx,
            log_trajectories=True,
            trajectory_log_prefix=log_prefix,
        )
    else:
        log_prefix = (
            cfg.policy_type + "_" + str(num_inference_steps) + "_" + cfg.task.task_name
        )

        runner_log = env_runner.run(
            policy,
            log_trajectories=True,
            trajectory_log_prefix=log_prefix,
        )

    env_name = cfg.task.task_name
    policy_type = cfg.policy_type
    filename = (
        policy_type + "_" + str(num_inference_steps) + "_" + env_name + "_actions.npy"
    )
    rewards_filename = (
        policy_type + "_" + str(num_inference_steps) + "_" + env_name + "_rewards.npy"
    )
    all_rewards_filename = (
        policy_type
        + "_"
        + str(num_inference_steps)
        + "_"
        + env_name
        + "_all_rewards.npy"
    )
    trajectories = np.load("trajectory_logs/" + filename)
    rewards = np.load("trajectory_logs/" + rewards_filename).squeeze()
    if env_name != "pusht_image":
        all_rewards = np.load("trajectory_logs/" + all_rewards_filename).squeeze()

    # Normalize_actions
    actions = torch.tensor(trajectories).to(device)
    normalized_actions = policy.normalizer["action"].normalize(actions)
    trajectories = normalized_actions.detach().cpu().numpy()

    trajectory_list = []
    print(rewards.shape)
    for i in range(trajectories.shape[0]):  # For each env trajectory
        if env_name != "pusht_image":
            if rewards[i] > 0:
                rollout_completed_idx = np.where(all_rewards[i] == 1)[0][0]
                trajectory_list.append(trajectories[i])  # [:rollout_completed_idx])
        else:
            trajectory_list.append(trajectories[i])

    dtw_variance = calculate_dtw_variance_from_dba(trajectory_list)
    if env_name != "pusht_image":
        success_rate = len(trajectory_list) / rewards.shape[0]
    else:
        success_rate = np.mean(rewards)
    print(f"Policy-> {policy_type}, Env-> {env_name}")
    print(f"DTW Variance: {dtw_variance}")
    print(f"Success rate: {success_rate}")

    # # dump log to json
    # Load existing logs if the file exists
    out_path = os.path.join(output_dir, "eval_log.json")
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            json_log_list = json.load(f)
    else:
        json_log_list = []

    # Append new result
    json_log_list.append(
        {
            "policy": policy_type,
            "task": env_name,
            "dtw_variance": dtw_variance,
            "success_rate": success_rate,
        }
    )

    # Write back
    with open(out_path, "w") as f:
        json.dump(json_log_list, f, indent=2)


if __name__ == "__main__":
    main()
