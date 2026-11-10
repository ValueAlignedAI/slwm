"""Causal instrumented tabletop and matched 10 Hz observations. No learned dependencies."""

import jax
import jax.numpy as jnp

AUDIO, VISION, PHYSICS, WORDS = 64, 768, 12, 48
ACTIONS, LENGTH = 7, 32
MOVES = jnp.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [0, 0], [0, 0]], jnp.int32)


def keys_for(ids, seed):
    return jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(seed), i))(ids)


def randint(keys, salt, shape, high):
    return jax.vmap(
        lambda k: jax.random.randint(jax.random.fold_in(k, salt), shape, 0, high)
    )(keys)


def initial(ids, seed):
    keys = keys_for(ids, seed)
    start = randint(keys, 0, (2,), 4)
    # Odd increments traverse all 16 cells without station collisions.
    cells = (
        randint(keys, 1, (1,), 16) + (1 + 2 * randint(keys, 2, (1,), 8)) * jnp.arange(3)
    ) % 16
    return dict(
        pos=start,
        stations=jnp.stack([cells % 4, cells // 4], axis=-1),
        goal=randint(keys, 3, (), 3),
        sign=2 * randint(keys, 4, (2,), 2) - 1,
        known=jnp.zeros(len(ids), bool),
        previous=jnp.zeros(len(ids), jnp.int32),
        tint=randint(keys, 5, (3,), 12).astype(jnp.float32) / 255,
    )


def target(state):
    return state["stations"][jnp.arange(len(state["goal"])), state["goal"]]


def expert(state):
    d = target(state) - state["pos"]
    axis = (jnp.abs(d[:, 1]) > jnp.abs(d[:, 0])).astype(jnp.int32)
    command = jnp.take_along_axis(d * state["sign"], axis[:, None], 1)[:, 0]
    move = 1 + 2 * axis + (command < 0)
    return jnp.where(
        jnp.all(d == 0, axis=1), 0, jnp.where(state["known"], move, 5)
    ).astype(jnp.int32)


def advance(state, action, tick, stall=False):
    delta = MOVES[action] * state["sign"]
    delta = jnp.where(
        jnp.asarray(stall) & (jnp.asarray(tick) % 5 == 2), jnp.zeros_like(delta), delta
    )
    return {
        **state,
        "pos": jnp.clip(state["pos"] + delta, 0, 3),
        "known": state["known"] | (action >= 5),
        "previous": action,
    }


def camera(stations, tint):
    # Physical station silhouettes occupy 3x3 native samples within each 4x4 board cell.
    gy, gx = jnp.mgrid[:16, :16]
    spots = (jnp.abs(gx[None, None] - 4 * stations[:, :, 0, None, None] - 1) <= 1) & (
        jnp.abs(gy[None, None] - 4 * stations[:, :, 1, None, None] - 1) <= 1
    )
    rgb = 0.07 + tint[:, None, None, :] + 0.80 * spots.transpose(0, 2, 3, 1)
    return (jnp.clip(rgb, 0, 1) * 255).astype(jnp.uint8).reshape(len(stations), VISION)


def observation(state, image, tick):
    b = len(image)
    last = state["previous"]
    tap = last >= 5
    polarity = jnp.where(last == 5, 1, -1)
    sensory_sign = polarity[:, None] * state["sign"]
    # Quantized four-channel log-band energies of a simulated contact resonance.
    band = jnp.arange(16)[None, None, :]
    centers = jnp.where(sensory_sign[:, 0] > 0, 10.0, 4.0)[:, None, None]
    channel = jnp.array([1.0, 0.8, 0.65, 0.55])[None, :, None]
    energy = 0.025 + tap[:, None, None] * channel * 0.84 * jnp.exp(
        -0.5 * ((band - centers) / 1.35) ** 2
    )
    a = (energy * 255).astype(jnp.uint8).reshape(b, AUDIO)
    n = jnp.zeros((b, PHYSICS), jnp.float32)
    n = n.at[:, :2].set(state["pos"].astype(jnp.float32) / 1.5 - 1)
    n = n.at[:, 2].set(tap * sensory_sign[:, 1])
    w = (
        jnp.zeros((b, 8), jnp.uint8)
        .at[:, 0]
        .set(4)
        .at[:, 1]
        .set(5)
        .at[:, 2]
        .set((state["goal"] + 1).astype(jnp.uint8))
    )
    return dict(a=a, v=image, n=n, w=w, u=last)


def sim_batch(ids, seed):
    keys = keys_for(ids, seed)
    state = initial(ids, seed)
    image = camera(state["stations"], state["tint"])
    free = randint(keys, 7, (LENGTH,), 7)
    delay = randint(keys, 8, (), 9)
    sign = 5 + randint(keys, 9, (), 2)
    # Independent behavior controls before and after a randomized calibration tap.
    actions = jnp.where(jnp.arange(LENGTH)[None, :] < delay[:, None], free % 5, free)
    actions = jnp.where(
        jnp.arange(LENGTH)[None, :] == delay[:, None], sign[:, None], actions
    )

    def step(st, item):
        t, act = item
        row = observation(st, image, t)
        row.update(policy=expert(st), plan=act)
        nxt = advance(st, act, t)
        obs = observation(nxt, image, t + 1)
        row.update(next_n=obs["n"], next_a=obs["a"])
        return nxt, row

    _, rows = jax.lax.scan(step, state, (jnp.arange(LENGTH), actions.T))
    return jax.tree.map(lambda value: value.swapaxes(0, 1), rows)


def tag_targets(words):
    """Multi-hot three goal colors and 13 annotated event classes."""
    classes = jnp.array([1, 2, 3, *range(32, 45)])
    return (words[..., None] == classes).any(-2).astype(jnp.float32)


def real_rows(bank, starts):
    ix = starts[:, None] + jnp.arange(LENGTH)[None, :]
    r = {k: bank[k][ix] for k in ("a", "v", "n", "w")}
    empty = jnp.zeros(ix.shape, jnp.int32)
    r.update(
        u=empty,
        policy=empty,
        plan=empty,
        next_n=bank["n"][ix + 1],
        next_a=bank["a"][ix + 1],
    )
    return r


def mix(bank, starts, step, seed, replay, count=128):
    ns = 3 * count // 4
    ids = step * ns + jnp.arange(ns, dtype=jnp.int32)
    if replay:
        ids = ids % 2048
    sim = sim_batch(ids, seed)
    rng = jax.random.fold_in(jax.random.PRNGKey(seed + 31), step)
    pick, mask = jax.random.split(rng)
    selected = starts[jax.random.randint(pick, (count - ns,), 0, len(starts))]
    real = real_rows(bank, selected)
    rows = jax.tree.map(
        lambda left, right: jnp.concatenate([left, right], axis=0), sim, real
    )
    # Exactly half in expectation see four simultaneous streams. Other rows hide one or more.
    modes = jax.random.randint(mask, (count,), 0, 16)
    patterns = jnp.array(
        [[1, 1, 1, 1]] * 8
        + [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        bool,
    )
    rows.update(keep=patterns[modes], is_sim=jnp.arange(count) < ns)
    return rows
