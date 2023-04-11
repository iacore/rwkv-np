`rwkv.py` is mostly copied from https://johanwind.github.io/2023/03/23/rwkv_details.html

You can install this by running `pip install -e .` inside this repo.

## Prepare the model

```
wget2 https://huggingface.co/BlinkDL/rwkv-4-pile-430m/resolve/main/RWKV-4-Pile-430M-20220808-8066.pth

# get .safetensors file
python convert.py RWKV-4-Pile-430M-20220808-8066.pth
```
