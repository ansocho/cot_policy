from tqdm import tqdm
from typing import Dict
import torch
import torch.nn.functional as F

from cot_policy.model.common.normalizer import LinearNormalizer
from cot_policy.policy.base_image_policy import BaseImagePolicy
from cot_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from cot_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from cot_policy.common.pytorch_util import dict_apply

from cot_policy.common.pca import PCA

from torchdiffeq import odeint

from torchcfm.optimal_transport import *
from torchcfm.utils import *
from tqdm import tqdm


class FMUnetImagePolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=4,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        sampling_method="euler",
        global_clusters=True,
        pca_features=100,
        num_clusters=64,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]

        low_dim_key = []
        rgb_key = []
        for key, value in shape_meta["obs"].items():
            if value.get("type") == "low_dim":
                low_dim_key.append(key)
            if value.get("type") == "rgb":
                rgb_key.append(key)
        self.lowdim_key = low_dim_key
        self.rgb_key = rgb_key
        print(self.rgb_key)

        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        self.num_pca_features = pca_features
        self.PCA = PCA(n_components=self.num_pca_features)

        ot_sampler = OTPlanSampler(method="exact")
        print("Using OT coupling")

        self.GLOBAL_CLUSTERS = global_clusters
        self.ot_sampler = ot_sampler
        self.obs_encoder = obs_encoder
        self.model = model
        self.num_clusters = num_clusters  # 32 for metaworld, 5 for robomimic

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps

        self.sampling_method = sampling_method

    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size,
        condition_data,
        local_cond=None,
        global_cond=None,
        global_cond_vae=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):

        noise = 1 * torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=None,
        )

        x = noise.detach().clone()

        t = torch.linspace(0, 1, steps=self.num_inference_steps).to(
            condition_data.device
        )

        x = odeint(
            lambda t, x: self.model(
                x,
                t * self.kwargs["pos_emb_scale"],
                local_cond=local_cond,
                global_cond=global_cond,
            ),
            x,
            t,
            method=self.sampling_method,
            atol=1e-3,
            rtol=1e-3,
        )[-1]

        return x

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(
                nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(
                nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            B,
            cond_data,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_train_tuple(self, z0, z1):
        t = torch.rand(z1.shape[0], 1, 1).to(z1.device)
        z_t = t * z1 + (1.0 - t) * z0
        target = z1 - z0
        return z_t, t, target

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(
                nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # Sample noise that we'll add to the images
        noise = 1 * torch.randn(trajectory.shape, device=trajectory.device)

        z0 = noise
        z1 = nactions
        if self.ot_sampler is not None:

            normalized_noise = z0.reshape(batch_size, -1)
            normalized_samples = z1.reshape(batch_size, -1)

            # conditions handling

            proprio = []
            for key in self.lowdim_key:
                proprio.append(this_nobs[key].reshape(batch_size, -1))
            proprio = torch.cat(proprio, dim=-1)
            image = this_nobs[self.rgb_key[0]].reshape(batch_size, -1)

            pca_features = self.PCA.transform(image, n_components=self.num_pca_features)

            clustered_global_cond = torch.cat([proprio, pca_features], dim=-1)

            normalized_noise_cond = clustered_global_cond[
                torch.randperm(clustered_global_cond.shape[0])
            ]

            eps = (
                1e1
                * torch.mean(torch.cdist(normalized_noise, normalized_samples))
                / torch.mean(torch.cdist(normalized_noise_cond, clustered_global_cond))
            )
            x0 = torch.cat([normalized_noise, eps * normalized_noise_cond], dim=-1)
            x1 = torch.cat([normalized_samples, eps * clustered_global_cond], dim=-1)

            x0, x1, _, global_cond = self.ot_sampler.sample_plan_with_labels(
                x0=x0,
                x1=x1,
                y1=global_cond,
            )

            z0 = x0[:, : normalized_noise.shape[1]].reshape(batch_size, horizon, -1)
            z1 = x1[:, : normalized_samples.shape[1]].reshape(batch_size, horizon, -1)

        x_t, t, target = self.get_train_tuple(z0=z0, z1=z1)

        # compute loss mask
        velocity_pred = self.model(
            x_t,
            t.squeeze() * self.kwargs["pos_emb_scale"],
            local_cond=local_cond,
            global_cond=global_cond,
        )

        error = (
            target.flatten(start_dim=1) - velocity_pred.flatten(start_dim=1)
        ).square()

        loss = error.mean()

        return loss

    def fit_pca(self, data_loader):
        all_data = []

        print("Fitting PCA...")
        # Iterate over the batches
        for batch in tqdm(data_loader, desc="Processing Batches", unit="batch"):
            batch_size = batch["action"].shape[0]

            # Extract the relevant information
            nobs = self.normalizer.normalize(batch["obs"])
            this_nobs = dict_apply(
                nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
            )
            image = this_nobs[self.rgb_key[0]].reshape(batch_size, -1)

            all_data.append(image)

        # Concatenate all processed batches
        all_data = torch.cat(all_data, dim=0)

        self.PCA.fit(all_data)

        return
