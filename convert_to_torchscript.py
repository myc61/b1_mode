#将leggedgym训练网络修改
import numpy as np
if not hasattr(np, "float"):
    np.float = float
from legged_gym.envs.b1.b1_config import B1RoughCfg, B1RoughCfgPPO
from rsl_rl.modules import ActorCritic

# 必须放在 import torch 之前
from isaacgym import gymapi, gymtorch  

import torch
from torch import nn

checkpoint_path = "logs/rough_b1/Sep03_13-25-42_/model_1500.pt"
torchscript_path = "deploy/train_mode/policy_torchscript.pt"
# ===============================
# 1. 读取 config
# ===============================
cfg = B1RoughCfg()
cfg_ppo = B1RoughCfgPPO()

obs_dim = cfg.env.num_observations
action_dim = cfg.env.num_actions

# 注意：这里的隐藏层必须和训练时一致
actor_hidden = [512, 256, 128]
critic_hidden = [512, 256, 128]
activation = 'elu'

# ===== 构建网络 =====
policy = ActorCritic(
    num_actor_obs=obs_dim,
    num_critic_obs=obs_dim,
    num_actions=action_dim,
    actor_hidden_dims=actor_hidden,
    critic_hidden_dims=critic_hidden,
    activation=activation
)

# ===== 加载权重 =====
print(f"Loading checkpoint from {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location="cpu")
policy.load_state_dict(ckpt['model_state_dict'])
policy.eval()
print("✅ Checkpoint loaded successfully")

# =====================
# 包装成一个可 trace 的推理模型
# =====================
class PolicyWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor):
        # act_inference 返回 (action, log_prob)
        action = self.policy.act_inference(obs)
        return action

wrapped_policy = PolicyWrapper(policy)

# =====================
# 生成 TorchScript
# =====================
example_input = torch.zeros(1, obs_dim)
scripted_policy = torch.jit.trace(wrapped_policy, example_input)
scripted_policy.save(torchscript_path)
print(f"✅ TorchScript policy saved to {torchscript_path}")
