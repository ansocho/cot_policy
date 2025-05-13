import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNet(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        obs_dim=0,
        hidden_sizes=(64, 64),
        nonlinearity="relu",  # either 'tanh' or 'relu'
        in_shift=None,
        in_scale=None,
        out_shift=None,
        out_scale=None,
    ):
        super(FCNet, self).__init__()

        self.input_dim = input_dim
        self.obs_dim = obs_dim
        self.output_dim = output_dim
        assert type(hidden_sizes) == tuple
        # We assume obs represents a vector embedding of the obsrvation
        self.layer_sizes = (input_dim + obs_dim,) + hidden_sizes + (output_dim,)

        # Batch Norm Layers
        # self.bn = torch.nn.BatchNorm1d(input_dim + obs_dim)

        # hidden layers
        self.fc_layers = nn.ModuleList(
            [
                nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1])
                for i in range(len(self.layer_sizes) - 1)
            ]
        )
        self.nonlinearity = torch.relu if nonlinearity == "relu" else torch.tanh

    def forward(self, x):

        # out = self.bn(x)
        out = x
        for i in range(len(self.fc_layers) - 1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)
        return out


class NODE(nn.Module):

    def __init__(
        self,
        state_net,
        V_net=None,
        equilibrium=None,
        output_net=None,
        epsilon=1e-3,
        alpha=1e-3,
        stable_node=False,
    ):

        super(NODE, self).__init__()

        self.epsilon = epsilon
        self.alpha = alpha
        self.state_net = state_net
        self.V_net = V_net
        if equilibrium is not None:
            self.equilibrium = equilibrium
        #        else:
        #            self.equilibrium = torch.zeros(
        #                state_net.layer_sizes[-1], dtype=torch.float32
        #            )
        if output_net is not None:
            self.reverse_output_net = output_net.backward
        else:
            self.reverse_output_net = nn.Identity()
        self.stable_node = stable_node

        # ted = 256
        # self.time_step_encoder = nn.Sequential(
        #     SinusoidalPosEmb(ted),
        #     nn.Linear(ted, ted * 4),
        #     nn.Mish(),
        #     nn.Linear(ted * 4, ted),
        # )

    def V(self, x):

        if isinstance(self.V_net, nn.ModuleList):
            V_value = 1
            for i, V in enumerate(self.V_net):
                equilibrium = self.reverse_output_net(self.equilibrium[i])
                V_value *= V(x - equilibrium.to(x.device))
            return V_value

        else:
            equilibrium = self.reverse_output_net(self.equilibrium)
            return self.V_net(x - equilibrium.to(x.device))

    def stabilizer(self, x, f):
        gradV_value = torch.autograd.grad(
            [a for a in self.V_value],
            [x],
            create_graph=True,
            only_inputs=True,
        )[0]

        sigmoid_scale = 10
        L = (f * gradV_value).sum(dim=-1).unsqueeze(dim=-1) + self.alpha * self.V_value
        D = (gradV_value**2).sum(dim=-1).unsqueeze(dim=-1) + self.epsilon
        stabilizing_vf = (
            -(
                gradV_value * (nn.ReLU()(L))
                + nn.Sigmoid()(sigmoid_scale * L) * self.epsilon * f
            )
            / D
        )

        return stabilizing_vf

    def forward(self, t, x, obs=None):

        # timesteps = t
        # if not torch.is_tensor(timesteps):
        #     # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
        #     timesteps = torch.tensor(
        #         [timesteps], dtype=torch.long, device=x.device
        #     )
        # elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
        #     timesteps = timesteps[None].to(x.device)
        # timesteps = timesteps.expand(x.shape[0])

        # t_embedding = self.time_step_encoder(timesteps)

        if obs is not None:
            # x = torch.cat([obs[..., int(t), :], x, t_embedding], dim=-1)
            # if not torch.all(x == 0):
            #     state = x / x.norm(dim=-1, keepdim=True)
            # else:
            #     state = x
            # x = x / x.norm(dim=-1, keepdim=True)
            x = F.normalize(x, p=2, dim=-1)
            o = F.normalize(obs[..., int(t), :], p=2, dim=-1)

            x = torch.cat([o, x], dim=-1)

        f = self.state_net(x)

        if self.stable_node:
            self.V_value = self.V(state)
            dx = f + self.stabilizer(state, f)
        else:
            dx = f

        return dx
