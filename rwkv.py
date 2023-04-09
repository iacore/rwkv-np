import numpy as np
import numba
from tqdm import tqdm
from tokenizers import Tokenizer

# def trace_shape(*args):
#     print("trace_shape", [arg.shape for arg in args])

jit = numba.jit(nopython=True, cache=True)

exp = np.exp
layer_norm = jit(lambda x, w, b: (x - np.mean(x)) / np.std(x) * w + b)
sigmoid = jit(lambda x: 1 / (1 + exp(-x)))


@jit
def time_mixing(
    x, last_x, last_num, last_den, decay, bonus, mix_k, mix_v, mix_r, Wk, Wv, Wr, Wout
):
    # trace_shape(x, last_x, last_num, last_den, decay, bonus, mix_k, mix_v, mix_r, Wk, Wv, Wr, Wout)

    k = Wk @ (x * mix_k + last_x * (1 - mix_k))
    v = Wv @ (x * mix_v + last_x * (1 - mix_v))
    r = Wr @ (x * mix_r + last_x * (1 - mix_r))

    wkv = (last_num + exp(bonus + k) * v) / (last_den + exp(bonus + k))
    rwkv = sigmoid(r) * wkv

    num = exp(-exp(decay)) * last_num + exp(k) * v
    den = exp(-exp(decay)) * last_den + exp(k)

    return Wout @ rwkv, (x, num, den)


@jit
def channel_mixing(x, last_x, mix_k, mix_r, Wk, Wr, Wv):
    k = Wk @ (x * mix_k + last_x * (1 - mix_k))
    r = Wr @ (x * mix_r + last_x * (1 - mix_r))
    vk = Wv @ np.maximum(k, 0) ** 2
    return sigmoid(r) * vk, x


from typing import NamedTuple


class Model(NamedTuple):
    tensors: dict[str, np.array]
    n_embd: int
    n_layer: int

    @staticmethod
    def load_safetensors(file: str) -> "Model":
        from safetensors import safe_open

        tensors_safe = safe_open(file, "numpy")
        tensors = {}
        for k in tqdm(tensors_safe.keys(), desc="Loading model"):
            tensors[k] = tensors_safe.get_tensor(k)
            if ".time_" in k:
                tensors[k] = tensors[k].squeeze()

        n_embd = int(tensors["ln_out.weight"].shape[0])
        n_layer = 0
        while f"blocks.{n_layer}.ln2.weight" in tensors:
            n_layer += 1

        return Model(tensors=tensors, n_embd=n_embd, n_layer=n_layer)


def RWKV(model: Model, token, state):
    tensors = model.tensors

    def p(key):
        return tensors[key]

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

    for i in range(model.n_layer):
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


if __name__ == "__main__":
    # converted using `convert.py` in this repo
    MODEL_FILE = "RWKV-4-Pile-430M-20220808-8066.safetensors"

    model = Model.load_safetensors(MODEL_FILE)

    # Available at https://github.com/BlinkDL/ChatRWKV/blob/main/20B_tokenizer.json
    tokenizer = Tokenizer.from_file("20B_tokenizer.json")

    prompt = "In a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."

    state = np.zeros((model.n_layer, 4, model.n_embd), dtype=np.float32)
    for token in tqdm(tokenizer.encode(prompt).ids, desc="Feeding prompt"):
        probs, state = RWKV(model, token, state)

    print(prompt, end="")
    while True:
        token = sample_probs(probs, temperature=1, top_p=0.8)
        word = tokenizer.decode([token])
        print(word, end="", flush=True)
        probs, state = RWKV(model, token, state)
