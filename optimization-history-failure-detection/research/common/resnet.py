"""Compact ResNet-18 for small RGB clinical images (MedMNIST-style 28×28)."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax

ResNetParams = dict[str, Any]
ApplyFn = Callable[[ResNetParams, jnp.ndarray], jnp.ndarray]

RESNET18_BLOCK_STRIDES = (1, 1, 2, 1, 2, 1, 2, 1)


def _conv(
    x: jnp.ndarray,
    w: jnp.ndarray,
    b: jnp.ndarray,
    *,
    stride: int = 1,
    padding: str | tuple[int, int] = "SAME",
) -> jnp.ndarray:
    if x.ndim == 3:
        dim_nums = ("HWC", "HWIO", "HWC")
    else:
        dim_nums = ("NHWC", "HWIO", "NHWC")
    out = jax.lax.conv_general_dilated(
        x,
        w,
        window_strides=(stride, stride),
        padding=padding,
        dimension_numbers=dim_nums,
    )
    return out + b


def _batch_norm(x: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    mean = jnp.mean(x, axis=(0, 1), keepdims=True)
    var = jnp.var(x, axis=(0, 1), keepdims=True)
    x_hat = (x - mean) / jnp.sqrt(var + 1e-5)
    return x_hat * scale + bias


def _relu(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.relu(x)


def _basic_block(
    x: jnp.ndarray,
    params: dict[str, jnp.ndarray],
    *,
    stride: int,
) -> jnp.ndarray:
    out = _conv(x, params["conv1_w"], params["conv1_b"], stride=stride)
    out = _batch_norm(out, params["bn1_scale"], params["bn1_bias"])
    out = _relu(out)
    out = _conv(out, params["conv2_w"], params["conv2_b"], stride=1)
    out = _batch_norm(out, params["bn2_scale"], params["bn2_bias"])

    if "shortcut_w" in params:
        shortcut = _conv(x, params["shortcut_w"], params["shortcut_b"], stride=stride)
        shortcut = _batch_norm(shortcut, params["shortcut_bn_scale"], params["shortcut_bn_bias"])
    else:
        shortcut = x
    return _relu(out + shortcut)


def _make_block_params(
    key: jax.Array,
    in_ch: int,
    out_ch: int,
    *,
    stride: int,
    downsample: bool,
) -> dict[str, jnp.ndarray]:
    k1, k2, k3 = jax.random.split(key, 3)
    params: dict[str, jnp.ndarray] = {
        "conv1_w": jax.random.normal(k1, (3, 3, in_ch, out_ch)) * jnp.sqrt(2.0 / (3 * 3 * in_ch)),
        "conv1_b": jnp.zeros((out_ch,), dtype=jnp.float32),
        "bn1_scale": jnp.ones((out_ch,), dtype=jnp.float32),
        "bn1_bias": jnp.zeros((out_ch,), dtype=jnp.float32),
        "conv2_w": jax.random.normal(k2, (3, 3, out_ch, out_ch)) * jnp.sqrt(2.0 / (3 * 3 * out_ch)),
        "conv2_b": jnp.zeros((out_ch,), dtype=jnp.float32),
        "bn2_scale": jnp.ones((out_ch,), dtype=jnp.float32),
        "bn2_bias": jnp.zeros((out_ch,), dtype=jnp.float32),
    }
    if downsample:
        params["shortcut_w"] = jax.random.normal(k3, (1, 1, in_ch, out_ch)) * jnp.sqrt(2.0 / in_ch)
        params["shortcut_b"] = jnp.zeros((out_ch,), dtype=jnp.float32)
        params["shortcut_bn_scale"] = jnp.ones((out_ch,), dtype=jnp.float32)
        params["shortcut_bn_bias"] = jnp.zeros((out_ch,), dtype=jnp.float32)
    return params


def init_resnet18_params(
    key: jax.Array,
    *,
    input_shape: tuple[int, int, int] = (28, 28, 3),
    num_classes: int = 9,
    base_width: int = 64,
) -> ResNetParams:
    """Initialize a ResNet-18 style network for small RGB inputs."""
    k0, *block_keys, k_fc = jax.random.split(key, 10)
    in_ch = input_shape[-1]

    stem_w = jax.random.normal(k0, (3, 3, in_ch, base_width)) * 0.01
    params: ResNetParams = {
        "stem_w": stem_w,
        "stem_b": jnp.zeros((base_width,), dtype=jnp.float32),
        "stem_bn_scale": jnp.ones((base_width,), dtype=jnp.float32),
        "stem_bn_bias": jnp.zeros((base_width,), dtype=jnp.float32),
        "fc_w": jax.random.normal(k_fc, (base_width * 8, num_classes)) * 0.01,
        "fc_b": jnp.zeros((num_classes,), dtype=jnp.float32),
    }

    channels = [base_width, base_width * 2, base_width * 4, base_width * 8]
    block_keys = list(block_keys)
    key_idx = 0
    prev = base_width
    for stage, out_ch in enumerate(channels):
        for block_idx in range(2):
            stride = 2 if block_idx == 0 and stage > 0 else 1
            downsample = prev != out_ch or stride > 1
            block_params = _make_block_params(
                block_keys[key_idx],
                prev,
                out_ch,
                stride=stride,
                downsample=downsample,
            )
            params[f"block_{key_idx}"] = block_params
            prev = out_ch
            key_idx += 1

    return params


def resnet18_apply(params: ResNetParams, x: jnp.ndarray) -> jnp.ndarray:
    """Forward pass. ``x`` may be (H,W,C), (N,H,W), or (N,H,W,C)."""
    if x.ndim == 3 and x.shape[-1] not in (1, 3):
        x = x[..., None]
        single = False
    else:
        single = x.ndim == 3
    if single:
        x = x[None, ...]
    if x.ndim == 4 and x.shape[-1] == 1:
        x = jnp.repeat(x, 3, axis=-1)
    if x.shape[-1] != 3 and x.shape[1] == 3:
        x = jnp.transpose(x, (0, 2, 3, 1))

    out = _conv(x, params["stem_w"], params["stem_b"], stride=1)
    out = _batch_norm(out, params["stem_bn_scale"], params["stem_bn_bias"])
    out = _relu(out)

    for i, stride in enumerate(RESNET18_BLOCK_STRIDES):
        block_params = params[f"block_{i}"]
        out = _basic_block(out, block_params, stride=int(stride))

    out = jnp.mean(out, axis=(1, 2))
    logits = out @ params["fc_w"] + params["fc_b"]
    return logits[0] if single else logits


def resnet18_features(params: ResNetParams, x: jnp.ndarray) -> jnp.ndarray:
    """Penultimate global-average-pooled representation h(x) before the FC layer."""
    if x.ndim == 3 and x.shape[-1] not in (1, 3):
        x = x[..., None]
        single = False
    else:
        single = x.ndim == 3
    if single:
        x = x[None, ...]
    if x.ndim == 4 and x.shape[-1] == 1:
        x = jnp.repeat(x, 3, axis=-1)
    if x.shape[-1] != 3 and x.shape[1] == 3:
        x = jnp.transpose(x, (0, 2, 3, 1))

    out = _conv(x, params["stem_w"], params["stem_b"], stride=1)
    out = _batch_norm(out, params["stem_bn_scale"], params["stem_bn_bias"])
    out = _relu(out)

    for i, stride in enumerate(RESNET18_BLOCK_STRIDES):
        block_params = params[f"block_{i}"]
        out = _basic_block(out, block_params, stride=int(stride))

    out = jnp.mean(out, axis=(1, 2))
    return out[0] if single else out


def resnet18_probs(params: ResNetParams, x: jnp.ndarray, *, eps: float = 1e-12) -> jnp.ndarray:
    logits = resnet18_apply(params, x)
    if logits.ndim == 1:
        logits = logits[None, :]
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(logits)
    return exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)


@jax.jit
def batch_loss(params: ResNetParams, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    logits = resnet18_apply(params, x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def loss_grad_fn(params: ResNetParams, x: jnp.ndarray, y: jnp.ndarray):
    return jax.value_and_grad(batch_loss)(params, x, y)


def example_grad(params: ResNetParams, x: jnp.ndarray, y: jnp.ndarray) -> ResNetParams:
    """Per-example gradient for MSA scoring."""

    def loss_fn(p: ResNetParams) -> jnp.ndarray:
        logits = resnet18_apply(p, x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y)

    return jax.grad(loss_fn)(params)
