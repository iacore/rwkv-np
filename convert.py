from torch import load as torch_load
from safetensors.torch import save_file

weights = torch_load("RWKV-4-Pile-430M-20220808-8066.pth", 'cpu')

for k in weights.keys():
    print(k, weights[k].size(), weights[k].dtype)
    weights[k] = weights[k].float()

save_file(weights, "RWKV-4-Pile-430M-20220808-8066.safetensors")

