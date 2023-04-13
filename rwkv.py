import logging
import numpy as np
import numba
from tqdm import tqdm
from tokenizers import Tokenizer


logger = logging.getLogger(name="rwkv")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler("__pycache__/rwkv.log", mode="w"))

def trace_shape(*args):
    logger.debug("trace_shape", [arg.shape for arg in args])


jit = numba.jit(nopython=True, cache=True)

exp = np.exp

# zero also works
eps_std = np.array(0.000009999999747378752, dtype=np.float32)

@jit
def layer_norm(x, w, b):
    xee2 = x - np.mean(x)
    x2 = np.sqrt(np.mean(xee2*xee2) + eps_std)
    return (xee2/x2) * w + b

sigmoid = jit(lambda x: 1 / (1 + exp(-x)))


@jit
def time_mixing(
    x, last_x, last_num, last_den, decay, bonus, mix_k, mix_v, mix_r, Wk, Wv, Wr, Wout
):
    # trace_shape(x, last_x, last_num, last_den, decay, bonus, mix_k, mix_v, mix_r, Wk, Wv, Wr, Wout)

    k = Wk @ (x * mix_k + last_x * (1 - mix_k))
    v = Wv @ (x * mix_v + last_x * (1 - mix_v))
    r = Wr @ (x * mix_r + last_x * (1 - mix_r))

    common_0 = exp(bonus + k)
    wkv = (last_num + common_0 * v) / (last_den + common_0)
    rwkv = sigmoid(r) * wkv

    common_1 = exp(-exp(decay))
    common_2 = exp(k)
    num = common_1 * last_num + common_2 * v
    den = common_1 * last_den + common_2

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
        from safetensors.numpy import load_file

        tensors = load_file(file)

        for k in tensors.keys():
            if ".time_" in k:
                tensors[k] = tensors[k].squeeze()
            tensors[k] = np.float32(tensors[k])

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

    logger.debug(f"token {token=}")

    x = p("emb.weight")[token]
    logger.debug(f"after emb {x=}")
    x = layer_norm(x, *p_ln("blocks.0.ln0"))
    logger.debug(f"after ln_in {x=}")

    for i in range(model.n_layer):
        x_ = layer_norm(x, *p_ln(f"blocks.{i}.ln1"))
        logger.debug(f"after block.ln1 {i=} {x_=}")
        dx, state[i][:3] = time_mixing(x_, *state[i][:3], *p_att(f"blocks.{i}.att"))
        x = x + dx
        logger.debug(f"after att {i=} {x=}")

        x_ = layer_norm(x, *p_ln(f"blocks.{i}.ln2"))
        logger.debug(f"after block.ln2 {i=} {x_=}")
        dx, state[i][3] = channel_mixing(x_, state[i][3], *p_ffn(f"blocks.{i}.ffn"))
        x = x + dx
        logger.debug(f"after ffn {i=} {x=}")

    x = layer_norm(x, *p_ln("ln_out"))
    logger.debug(f"after ln_out {x=}")
    x = p("head.weight") @ x
    logger.debug(f"after head {x=}")

    e_x = exp(x - np.max(x))
    probs = e_x / e_x.sum()  # Softmax of x
    
    minids = np.argsort(probs, axis=0)
    L = minids.shape[0]
    max10_ids = []
    for i, x in enumerate(minids):
        if x > L - 10:
            max10_ids.append(i)

    logger.debug(f"after softmax {probs=} max10_ids={max10_ids}")

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
    import argparse, shelve

    parser = argparse.ArgumentParser(prog="rwkv.py")
    parser.add_argument("model_file", help="model file in safetensors format")
    parser.add_argument("--no-cache", action='store_true', help="disable cache")
    args = parser.parse_args()

    model = Model.load_safetensors(args.model_file)

    # Available at https://github.com/BlinkDL/ChatRWKV/blob/main/20B_tokenizer.json
    tokenizer = Tokenizer.from_file("20B_tokenizer.json")

    prompt = "In a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."

    probs, state = None, None
    
    def feed_prompt(prompt: str):
        global probs, state
        state = np.zeros((model.n_layer, 4, model.n_embd), dtype=np.float32)
        for token in tqdm(tokenizer.encode(prompt).ids, desc="Feeding prompt"):
            probs, state = RWKV(model, token, state)

    if args.no_cache:
        feed_prompt(prompt)
    else:
        with shelve.open(
            f"__pycache__/rwkv.{model.n_embd}-{model.n_layer}.prompt_cache.shelf"
        ) as db:
            try:
                probs, state = db[prompt]
            except KeyError:
                feed_prompt(prompt)
                db[prompt] = probs, state

    print(prompt, end="")
    while True:
        token = sample_probs(probs, temperature=1, top_p=0.8)
        word = tokenizer.decode([token])
        print(word, end="", flush=True)
        probs, state = RWKV(model, token, state)
