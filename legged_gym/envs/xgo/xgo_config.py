# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class XGORoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.6] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'lf_hip_joint': 0.1,
            'rf_hip_joint': -0.1,
            'lh_hip_joint': 0.1,
            'rh_hip_joint': -0.1,
            'lf_upper_leg_joint': 0.8,
            'rf_upper_leg_joint': 0.8,
            'lh_upper_leg_joint': 1,
            'rh_upper_leg_joint': 1,
            'lf_lower_leg_joint': -1.4,
            'rf_lower_leg_joint': -1.4,
            'lh_lower_leg_joint': -1.4,
            'rh_lower_leg_joint': -1.4,
        }
    class noise( LeggedRobotCfg.noise ):
        class noise_scales( LeggedRobotCfg.noise.noise_scales ):
            target_dist_noise_scale = 0.1
    class env(LeggedRobotCfg.env):
        num_actions = 12  # B1为12自由度
        #num_observations = 237
        episode_length_s = 25  # 每个episode持续25秒
        target_pos = [8.0, 0.0, 0.0]  # B1的目标点（x=8m、y=0m、z=0m）    
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 50.}  # [N*m/rad]
        damping = {'joint': 5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/yahboom_description/urdf/xgo_rviz.urdf'
        name = "b1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 2.0         # 线速度跟踪奖励，鼓励按目标速度运动
            tracking_ang_vel = 1.0         # 角速度跟踪奖励，鼓励按目标角速度运动
            feet_air_time = 1.0            # 抬腿奖励，鼓励步态
            torques = -0.0002              # 扭矩惩罚，防止动作过大
            dof_pos_limits = -10.0         # 关节超限惩罚
            collision = -2.0               # 碰撞惩罚，防止摔倒
            action_rate = -0.01            # 动作变化惩罚，鼓励平滑
            lin_vel_z = -2.0               # z轴速度惩罚，防止跳跃
            ang_vel_xy = -0.05             # xy轴角速度惩罚，防止剧烈旋转
class XGORoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_xgo'

  