from safetensors import safe_open

weights = safe_open("RWKV-4-Pile-430M-20220808-8066.safetensors", "numpy")

if __name__ == '__main__':
    for k in weights.keys():
        a = weights.get_tensor(k)
        print(k, a.shape, a.dtype)
