import numpy as np
import re
from scipy.spatial.transform import Rotation as R

def parse_inertia_line(line: str):
    """从URDF inertia行里提取惯性张量"""
    # 正则提取浮点数
    pattern = r'(\w+)="([-+]?[\d\.eE+-]+)"'
    values = dict(re.findall(pattern, line))

    ixx = float(values['ixx'])
    iyy = float(values['iyy'])
    izz = float(values['izz'])
    ixy = float(values['ixy'])
    ixz = float(values['ixz'])
    iyz = float(values['iyz'])

    I = np.array([
        [ixx, ixy, ixz],
        [ixy, iyy, iyz],
        [ixz, iyz, izz]
    ])
    return I

def inertia_to_diaginertia_quat(I):
    """把惯性张量转换为MJCF格式的diaginertia和quat"""
    eigvals, eigvecs = np.linalg.eigh(I)

    # 排序：从小到大
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 确保旋转矩阵是右手系（det=+1）
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 0] *= -1

    # 转成四元数 (MJCF用 w x y z)
    quat = R.from_matrix(eigvecs).as_quat()
    quat = np.roll(quat, 1)  # xyzw → wxyz

    return eigvals, quat

# 示例
line = '<inertia ixx="0.006341369" ixy="-3e-09" ixz="-8.7951e-05" iyy="0.006355157" iyz="-1.336e-06" izz="3.9188e-05"/>'

I = parse_inertia_line(line)
eigvals, quat = inertia_to_diaginertia_quat(I)

print("diaginertia =", eigvals)
print("quat =", quat)

