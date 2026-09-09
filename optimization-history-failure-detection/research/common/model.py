"""Shared Adam MLP classifier used as the Phase-0 baseline."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

HIDDEN_DIM = 256


def init_mlp_params(
    key: jax.Array,
    input_dim: int = 784,
    hidden: int = HIDDEN_DIM,
    num_classes: int = 10,
) -> dict[str, jnp.ndarray]:
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "w1": jax.random.normal(k1, (input_dim, hidden)) * 0.01,
        "b1": jnp.zeros((hidden,), dtype=jnp.float32),
        "w2": jax.random.normal(k2, (hidden, hidden)) * 0.01,
        "b2": jnp.zeros((hidden,), dtype=jnp.float32),
        "w3": jax.random.normal(k3, (hidden, num_classes)) * 0.01,
        "b3": jnp.zeros((num_classes,), dtype=jnp.float32),
    }


def mlp_apply(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    x = x.reshape((x.shape[0], -1))
    h1 = jax.nn.relu(x @ params["w1"] + params["b1"])
    h2 = jax.nn.relu(h1 @ params["w2"] + params["b2"])
    return h2 @ params["w3"] + params["b3"]


@jax.jit
def batch_loss(params: dict[str, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    logits = mlp_apply(params, x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def loss_grad_fn(params, x, y):
    return jax.value_and_grad(batch_loss)(params, x, y)


def adam_optimizer(learning_rate: float) -> optax.GradientTransformation:
    return optax.adam(learning_rate)
