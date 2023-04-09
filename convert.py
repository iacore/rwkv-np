import sys
from torch import load as torch_load
from safetensors.torch import save_file

file = sys.argv[1]
filebase, _ = file.rsplit(".", maxsplit=1)

weights = torch_load(file, "cpu")

for k in weights.keys():
    print(k, weights[k].size(), weights[k].dtype)
    weights[k] = weights[k].float()

save_file(weights, f"{filebase}.safetensors")
