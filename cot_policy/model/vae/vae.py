import torch
import torch.nn as nn
import torch.nn.functional as F

# from tqdm.auto import tqdm
import wandb
import numpy as np


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, cond_dim=0):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim + cond_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, output_dim)

    def forward(self, z, cond):
        z = F.normalize(z, p=2, dim=-1)
        cond = F.normalize(cond, p=2, dim=-1)
        z = torch.cat([z, cond], dim=-1)  # Concatenate latent and conditioning vector
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = self.linear3(z)  # No activation for continuous output
        return z


# class Decoder(nn.Module):
#     def __init__(self, latent_dim, output_dim, cond_dim=0):
#         super(Decoder, self).__init__()

#         mean_net = nn.Sequential(
#             nn.Linear(latent_dim + cond_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, output_dim)
#         )

#         log_var_net = nn.Sequential(
#             nn.Linear(latent_dim + cond_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, output_dim)
#         )

#         self.mean_net = mean_net
#         self.log_var_net = log_var_net

#     def forward(self, z, cond=None):
#         if cond is not None:
#             z = F.normalize(z, p=2, dim=-1)
#             cond = F.normalize(cond, p=2, dim=-1)
#             z = torch.cat([z, cond], dim=-1)  # Concatenate latent and conditioning vector
#         mu = self.mean_net(z)
#         log_sigma2 = self.log_var_net(z)
#         return mu, log_sigma2


class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, cond_dim=0):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim + cond_dim, 512)
        self.linear2 = nn.Linear(512, latent_dim)
        self.linear3 = nn.Linear(512, latent_dim)

        self.kl = 0

    def forward(self, x, cond=None):
        x = torch.flatten(x, start_dim=1)
        if cond is not None:
            x = F.normalize(x, p=2, dim=-1)
            cond = F.normalize(cond, p=2, dim=-1)
            x = torch.cat([x, cond], dim=-1)  # Concatenate input and conditioning vecto
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))

        epsilon = torch.randn_like(sigma).to(x.device)
        z = mu + sigma * epsilon

        # KL Divergence
        self.kl = 0.5 * torch.sum(sigma**2 + mu**2 - torch.log(sigma**2) - 1)
        return z


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, device):
        super(VAE, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.encoder = VariationalEncoder(self.input_dim, self.latent_dim)
        self.decoder = Decoder(self.latent_dim, self.input_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def sample(self, batch_size):
        # with torch.no_grad():
        #     z = torch.randn((batch_size, self.latent_dim)).to(
        #         device=self.device, dtype=torch.float32
        #     )  # Sample from prior
        #     mu = self.decoder(z)
        #     sampled_x = mu + torch.randn_like(mu)
        #     return sampled_x

        with torch.no_grad():
            # Sample latent variable from standard normal distribution
            z = torch.randn((batch_size, self.latent_dim)).to(
                device=cond.device, dtype=torch.float32
            )
            # Get predicted mean and log variance from the decoder
            mu, log_sigma2 = self.decoder(z)
            sigma = torch.exp(
                0.5 * log_sigma2
            )  # Convert log variance to standard deviation

            # Sample from the predicted Gaussian (mu, sigma^2)
            sampled_x = mu + sigma * torch.randn_like(mu)  # Reparametrize
            sampled_x = sampled_x.reshape(batch_size, self.horizon, -1)
            return sampled_x


class cVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim, horizon):
        super(cVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.horizon = horizon
        self.encoder = VariationalEncoder(
            self.input_dim, self.latent_dim, self.cond_dim
        )
        self.decoder = Decoder(self.latent_dim, self.input_dim, self.cond_dim)

    def forward(self, x, cond):
        z = self.encoder(x, cond)
        return self.decoder(z, cond)

    def sample(self, batch_size, cond):
        with torch.no_grad():
            z = torch.randn((batch_size, self.latent_dim)).to(
                device=cond.device, dtype=torch.float32
            )  # Sample from prior
            mu = self.decoder(z, cond)
            sampled_x = mu + 0.3 * torch.randn_like(mu)
            sampled_x = sampled_x.reshape(batch_size, self.horizon, -1)
            return sampled_x

        # with torch.no_grad():
        #     # Sample latent variable from standard normal distribution
        #     z = torch.randn((batch_size, self.latent_dim)).to(
        #         device=cond.device, dtype=torch.float32
        #     )
        #     # Get predicted mean and log variance from the decoder
        #     mu, log_sigma2 = self.decoder(z, cond)
        #     sigma = torch.exp(0.5 * log_sigma2)  # Convert log variance to standard deviation

        #     # Sample from the predicted Gaussian (mu, sigma^2)
        #     sampled_x = mu + sigma * torch.randn_like(mu)  # Reparametrize
        #     sampled_x = sampled_x.reshape(batch_size, self.horizon, -1)
        #     return sampled_x
