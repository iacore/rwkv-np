import numpy as np
from tokenizers import Tokenizer
from safetensors import safe_open

layer_norm = lambda x, w, b: (x - np.mean(x)) / np.std(x) * w + b
exp = np.exp
sigmoid = lambda x: 1 / (1 + exp(-x))


def time_mixing(
    x, last_x, last_num, last_den, decay, bonus, mix_k, mix_v, mix_r, Wk, Wv, Wr, Wout
):
    k = Wk @ (x * mix_k + last_x * (1 - mix_k))
    v = Wv @ (x * mix_v + last_x * (1 - mix_v))
    r = Wr @ (x * mix_r + last_x * (1 - mix_r))

    wkv = (last_num + exp(bonus + k) * v) / (last_den + exp(bonus + k))
    rwkv = sigmoid(r) * wkv

    num = exp(-exp(decay)) * last_num + exp(k) * v
    den = exp(-exp(decay)) * last_den + exp(k)

    return Wout @ rwkv, (x, num, den)


def channel_mixing(x, last_x, mix_k, mix_r, Wk, Wr, Wv):
    k = Wk @ (x * mix_k + last_x * (1 - mix_k))
    r = Wr @ (x * mix_r + last_x * (1 - mix_r))
    vk = Wv @ np.maximum(k, 0) ** 2
    return sigmoid(r) * vk, x


def RWKV(model, token, state):
    def p(key):
        return model[key]

    def p_ln(prefix):
        return [
            p(key)
            for key in [
                prefix + ".weight",
                prefix + ".bias",
            ]
        ]

    def p_att(prefix):
        return [
            p(key)
            for key in [
                prefix + ".time_decay",
                prefix + ".time_first",
                prefix + ".time_mix_k",
                prefix + ".time_mix_v",
                prefix + ".time_mix_r",
                prefix + ".key.weight",
                prefix + ".value.weight",
                prefix + ".receptance.weight",
                prefix + ".output.weight",
            ]
        ]

    def p_ffn(prefix):
        return [
            p(key)
            for key in [
                prefix + ".time_mix_k",
                prefix + ".time_mix_r",
                prefix + ".key.weight",
                prefix + ".receptance.weight",
                prefix + ".value.weight",
            ]
        ]

    x = p("emb.weight")[token]
    x = layer_norm(x, *p_ln("blocks.0.ln0"))

    for i in range(N_LAYER):
        x_ = layer_norm(x, *p_ln(f"blocks.{i}.ln1"))
        dx, state[i][:3] = time_mixing(x_, *state[i][:3], *p_att(f"blocks.{i}.att"))
        x = x + dx

        x_ = layer_norm(x, *p_ln(f"blocks.{i}.ln2"))
        dx, state[i][3] = channel_mixing(x_, state[i][3], *p_ffn(f"blocks.{i}.ffn"))
        x = x + dx

    x = layer_norm(x, *p_ln("ln_out"))
    x = p("head.weight") @ x

    e_x = exp(x - np.max(x))
    probs = e_x / e_x.sum()  # Softmax of x

    return probs, state


##########################################################################################################


def sample_probs(probs, temperature=1.0, top_p=0.85):
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = sorted_probs[np.argmax(cumulative_probs > top_p)]
    probs[probs < cutoff] = 0
    probs = probs ** (1 / temperature)
    return np.random.choice(a=len(probs), p=probs / np.sum(probs))


# converted using `convert.py` in this repo
MODEL_FILE = 'RWKV-4-Pile-430M-20220808-8066.safetensors'
N_LAYER = 24
N_EMBD = 1024

print(f'\nLoading {MODEL_FILE}')
weights_safe = safe_open(MODEL_FILE, "numpy")
weights = {}
for k in weights_safe.keys():
    weights[k] = weights_safe.get_tensor(k)
    if '.time_' in k: weights[k] = weights[k].squeeze()


# Available at https://github.com/BlinkDL/ChatRWKV/blob/main/20B_tokenizer.json
tokenizer = Tokenizer.from_file("20B_tokenizer.json")

print()
print(f"Preprocessing context")

context = "In a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."

state = np.zeros((N_LAYER, 4, N_EMBD), dtype=np.float32)
for token in tokenizer.encode(context).ids:
    probs, state = RWKV(weights, token, state)

print(context, end="")
for i in range(100):
    token = sample_probs(probs)
    print(tokenizer.decode([token]), end="", flush=True)
    probs, state = RWKV(weights, token, state)
