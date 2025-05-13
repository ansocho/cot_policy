import os
import sys
from typing import Dict
import numpy as np

sys.path.append("/home/soho/workspace/ot_policy")
import diffusion_policy.common.metaworld_utils.file_utils as FileUtils
import diffusion_policy.common.metaworld_utils.obs_utils as ObsUtils
from diffusion_policy.common.metaworld_utils.sequence_dataset import SequenceDataset
from diffusion_policy.common.sampler import (
    get_val_mask,
    downsample_mask,
)
import copy
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
import torch


def build_dataset(
    data_prefix,
    suite_name,
    benchmark_name,
    task_name,
    mode,
    seq_len,
    frame_stack,
    shape_meta,
    extra_obs_modality=None,
    obs_seq_len=1,
    lowdim_obs_seq_len=None,
    load_obs=True,
    n_demos=None,
    load_next_obs=False,
    dataset_keys=("actions",),
):
    n_tasks = len(task_name)

    obs_modality = {
        "rgb": list(shape_meta["observation"]["rgb"].keys()),
        "low_dim": list(shape_meta["observation"]["lowdim"].keys()),
    }
    if extra_obs_modality is not None:
        for key in extra_obs_modality:
            obs_modality[key] = obs_modality[key] + extra_obs_modality[key]

    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})
    # currently we assume tasks from same benchmark have the same shape_meta
    dataset_args = get_task_dataset(
        dataset_path=os.path.join(
            data_prefix, suite_name, benchmark_name, mode, f"{task_name}.hdf5"
        ),
        obs_modality=obs_modality,
        seq_len=seq_len,
        obs_seq_len=obs_seq_len,
        lowdim_obs_seq_len=lowdim_obs_seq_len,
        load_obs=load_obs,
        frame_stack=frame_stack,
        n_demos=n_demos,
        load_next_obs=load_next_obs,
        dataset_keys=dataset_keys,
    )

    return dataset_args


def get_task_dataset(
    dataset_path,
    obs_modality,
    seq_len=1,
    obs_seq_len=1,
    lowdim_obs_seq_len=None,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",
    few_demos=None,
    load_obs=True,
    n_demos=None,
    load_next_obs=False,
    dataset_keys=None,
):
    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )
    seq_len = seq_len
    filter_key = filter_key
    if load_obs:
        obs_keys = shape_meta["all_obs_keys"]
    else:
        obs_keys = []

    if dataset_keys is None:
        dataset_keys = [
            "actions",
        ]

    sequence_dataset_args = {
        "hdf5_path": dataset_path,
        "obs_keys": obs_keys,
        "dataset_keys": dataset_keys,
        "load_next_obs": load_next_obs,
        "frame_stack": frame_stack,
        "seq_length": seq_len,
        "obs_seq_length": obs_seq_len,
        "lowdim_obs_seq_length": lowdim_obs_seq_len,
        "pad_frame_stack": True,
        "pad_seq_length": True,
        "get_pad_mask": False,
        "hdf5_cache_mode": hdf5_cache_mode,
        "hdf5_use_swmr": False,
        "filter_by_attribute": filter_key,
        "few_demos": few_demos,
        "n_demos": n_demos,
    }

    return sequence_dataset_args


class MetaworldImageDataset(BaseImageDataset):
    def __init__(
        self,
        data_prefix,
        suite_name,
        benchmark_name,
        task_name,
        mode,
        seq_len,
        frame_stack,
        shape_meta,
        extra_obs_modality=None,
        obs_seq_len=1,
        lowdim_obs_seq_len=None,
        load_obs=True,
        n_demos=None,
        load_next_obs=False,
        dataset_keys=("actions",),
        val_ratio=0.0,
        max_train_episodes=None,
        seed=42,
    ):
        super().__init__()
        dataset_args = build_dataset(
            data_prefix=data_prefix,
            suite_name=suite_name,
            benchmark_name=benchmark_name,
            task_name=task_name,
            mode=mode,
            seq_len=seq_len,
            frame_stack=frame_stack,
            shape_meta=shape_meta,
            extra_obs_modality=extra_obs_modality,
            obs_seq_len=obs_seq_len,
            lowdim_obs_seq_len=lowdim_obs_seq_len,
            load_obs=load_obs,
            n_demos=n_demos,
            load_next_obs=load_next_obs,
            dataset_keys=dataset_keys,
        )

        val_mask = get_val_mask(n_episodes=n_demos, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed
        )

        self.sequence_dataset = SequenceDataset(**dataset_args, episode_mask=train_mask)

        self.n_demos = self.sequence_dataset.n_demos
        n_sequences = self.sequence_dataset.total_num_sequences

        self.train_mask = train_mask
        self.dataset_args = dataset_args

        print("\n===================  Benchmark Information  ===================")
        print(f" Name: MetaWorld")
        print(f" # Task: {task_name}")
        print(f" # demonstrations: {self.n_demos}")
        print(f" # sequences: {n_sequences}")
        print(
            "=======================================================================\n"
        )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sequence_dataset = SequenceDataset(
            **self.dataset_args, episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def __len__(self) -> int:
        return len(self.sequence_dataset)

    def get_normalizer(self, mode="limits", **kwargs):
        data = self.sequence_dataset.get_data()
        data = {
            "action": data["actions"],
            "robot_states": data["robot_states"][..., :4],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["corner_rgb"] = get_image_range_normalizer()
        return normalizer

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sequence_dataset[idx]
        sample["obs"]["corner_rgb"] = (
            sample["obs"]["corner_rgb"].astype(np.float32) / 255.0
        )
        sample["obs"]["robot_states"] = sample["obs"]["robot_states"][..., :4]
        sample["action"] = sample["actions"]
        return sample


if __name__ == "__main__":

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    config_dict = {
        "data_prefix": "/media/storage/soho/data",
        "suite_name": "metaworld",
        "benchmark_name": "ML1",
        "task_name": "soccer-v2",
        "mode": "train",
        "seq_len": 4,
        "frame_stack": 1,
        "obs_seq_len": 2,
        "lowdim_obs_seq_len": None,
        "shape_meta": {
            "action_dim": 4,
            "observation": {
                "rgb": {"corner_rgb": [3, "${task.img_height}", "${task.img_width}"]},
                "lowdim": {"robot_states": 8},
            },
            "task": {"type": "onehot", "n_tasks": 1},
        },
        "load_obs": True,
        "n_demos": 205,
        "load_next_obs": False,
        "val_ratio": 0.1,
        "max_train_episodes": 200,
    }

    dataset = MetaworldImageDataset(**config_dict)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    n_components = 2

    all_data = []

    # Collect and flatten data
    for batch in dataloader:
        if isinstance(
            batch, (tuple, list, dict)
        ):  # If __getitem__ returns tuple or list
            batch = batch["actions"]  # Assuming features are in the first element
        all_data.append(batch.numpy() if hasattr(batch, "numpy") else batch)

    # Flatten the data
    all_data = np.vstack(all_data)
    all_data_flattened = all_data.reshape(len(all_data), -1)

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(all_data_flattened)

    output_path = "pca_scatter_plot.png"
    # Create PCA scatter plot and save to file
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.7, s=10)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Scatter Plot")
    plt.grid(True)
    plt.savefig(output_path, dpi=300)
    plt.close()  # Close the plot to free memory
