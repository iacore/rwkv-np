from torch import load as torch_load

weights = torch_load("RWKV-4-Pile-430M-20220808-8066.pth", 'cpu')

if __name__ == '__main__':
    for k in weights.keys():
        a = weights[k].float().numpy()
        print(k, a.shape, a.dtype)
