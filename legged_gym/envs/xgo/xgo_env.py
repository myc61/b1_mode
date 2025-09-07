# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Adapted for Unitree B1 Robot

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import torch
from torch import Tensor
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
from isaacgym import gymtorch

class B1Robot(LeggedRobot):
    """宇树B1四足机器人环境类，继承自LeggedRobot基础类，适配B1硬件特性与自定义任务需求"""
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        # 1. 调用父类构造函数，初始化基础仿真环境
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        
        # 3. B1关节映射验证（确保与URDF关节名匹配，12自由度：4条腿×3关节/腿）
        self.default_dof_pos = torch.tensor(
            [self.cfg.init_state.default_joint_angles[name] for name in self.dof_names],
            device=self.device
        )
        print("关节顺序:", self.dof_names)
        print("映射后的初始关节角度:", self.default_dof_pos.cpu().numpy())
        self._verify_b1_joints()

    def _verify_b1_joints(self):
        """验证B1关节数量与命名是否正确（防止URDF加载错误）"""
        expected_joint_count = 12  # B1为12自由度机器人（hip_yaw, hip_roll, hip_pitch, thigh, calf×4腿）
        print("elf.num_dof:",self.num_dof)
        if self.num_dof != expected_joint_count:
            raise ValueError(f"B1机器人需12个自由度，当前加载{self.num_dof}个关节，请检查URDF文件！")
        # 验证关键关节名（匹配宇树B1 URDF关节命名规范）
        required_joint_keywords = ["hip", "thigh", "calf"]
        for joint_name in self.dof_names:
            if not any(kw in joint_name.lower() for kw in required_joint_keywords):
                raise ValueError(f"关节名{joint_name}不匹配B1规范，需包含'hip'/'thigh'/'calf'")

    def _reward_arrive_target_point(self):
        current_base_xy = self.root_states[:, :2]
        target_xy = self.target_pos[:2].unsqueeze(0).repeat(current_base_xy.shape[0], 1)
        dist_to_target = torch.norm(current_base_xy - target_xy, dim=1)
        
        max_reward = 10.0
        min_dist = 0.2
        max_dist = 15.0
        
        return torch.where(
            dist_to_target <= min_dist,
            torch.full_like(dist_to_target, max_reward),
            torch.clamp(max_reward * (1 - dist_to_target / max_dist), min=0.0)
        )

    def _reward_total_distance(self):
        """
        仅返回累计移动距离（纯数值），框架会自动乘以配置的 total_distance 权重
        无需手动乘权重，避免重复计算
        """
        # 1. 获取当前/上一步底座x/y坐标（忽略z轴）
        current_base_xy = self.root_states[:, :2].clone()
        prev_base_xy = self.prev_base_pos[:, :2]
        
        # 2. 计算单步距离（欧氏距离，非负）
        step_distance = torch.norm(current_base_xy - prev_base_xy, dim=1)
        
        # 3. 累加总距离（过滤微小抖动）
        self.total_distance += torch.clamp(step_distance, min=0.0)
        
        # 4. 更新上一步位置
        self.prev_base_pos[:, :2] = current_base_xy
        
        # 关键：仅返回累计距离（纯数值），权重由框架自动乘
        return self.total_distance


    def check_termination(self):
        """扩展终止条件：在基础条件中添加「到达目标点终止」，明确任务完成标志"""
        # 1. 调用父类方法，检查基础终止条件（跌倒、超时等）
        super().check_termination()
        
        # # 2. 新增：到达目标点终止（距离<0.2m时，终止当前episode）
        # current_base_xy = self.root_states[:, :2]
        # target_xy = self.target_pos[:2].unsqueeze(0).repeat(current_base_xy.shape[0], 1)
        # dist_to_target = torch.norm(current_base_xy - target_xy, dim=1)
        
        # # 将「到达目标」加入终止缓冲区（True表示需要重置环境）
        # self.reset_buf |= (dist_to_target < 0.2)

    def _compute_torques(self, actions):
        """适配B1关节特性的扭矩计算：继承PD位置控制，确保关节力符合B1硬件限制"""
        # 1. 调用父类PD控制逻辑（B1默认使用位置控制，与基础类一致）
        torques = super()._compute_torques(actions)
        
        # 2. B1特有：限制髋关节扭矩（避免过大扭矩损坏关节，根据B1硬件参数调整）
        hip_joint_indices = [i for i, name in enumerate(self.dof_names) if "hip" in name.lower()]
        if hip_joint_indices:
            hip_torque_limit = torch.tensor(250.0, device=self.device)  # B1髋关节最大扭矩约250N·m
            torques[:, hip_joint_indices] = torch.clip(
                torques[:, hip_joint_indices],
                -hip_torque_limit,
                hip_torque_limit
            )
        
        return torques


    def _reset_dofs(self, env_ids):
            """重写：直接用标准初始关节角度，无随机扰动"""
            self.dof_pos[env_ids] = self.default_dof_pos.repeat(len(env_ids), 1)
            self.dof_vel[env_ids] = 0.

            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def reset(self):
        """重置环境时，初始化位置和累计距离"""
        obs, privileged_obs = super().reset()
        # 记录初始位置和上一步位置
        # self.initial_base_pos = self.root_states[:, :3].clone()  # 保存初始x/y/z坐标
        # self.prev_base_pos = self.root_states[:, :3].clone()      # 上一步位置初始化为初始位置
        # self.total_distance.zero_()  # 重置累计距离
        print("关节顺序:", self.dof_names)
        print("初始关节角度:", self.dof_pos[0].cpu().numpy())  # 打印第一个环境的关节角度
        return obs, privileged_obs