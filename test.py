from scipy.spatial.transform import Rotation as R
from lietorch import SO3
import torch

# q = torch.randn(1, 4)
q = torch.tensor([0,0,0,1]).float()
q = q / q.norm(dim=-1, keepdim=True)

r = R.from_quat(q)

print(r.as_matrix())


litR = SO3.InitFromVec(q)

print(litR.matrix())
