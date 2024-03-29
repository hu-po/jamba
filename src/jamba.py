import argparse
import time
import itertools

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax import jit, grad, random
from jax.example_libraries import optimizers
import numpy.random as npr

from utils import check_gpu
from mnist import mnist


parser = argparse.ArgumentParser()
# training
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)

# directories and paths
parser.add_argument("--run_name", type=str, default="test1")
parser.add_argument("--data_dir", type=str, default="/home/oop/dev/data/mnist")
parser.add_argument("--ckpt_dir", type=str, default="/ckpt")
parser.add_argument("--save_ckpt", type=bool, default=False)
parser.add_argument("--logs_dir", type=str, default="/logs")

# learning rate and betas from paper Section E.2
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--b1", type=float, default=0.9)
parser.add_argument("--b2", type=float, default=0.95)

# Mamba model hyperparameters
parser.add_argument("--num_blocks", type=int, default=2)  # number of mamba blocks
parser.add_argument("--dim_h", type=int, default=4)  # dimmension of hidden state h
parser.add_argument("--dim_Δ", type=int, default=1)  # dimmension of Δ
parser.add_argument("--dim_conv", type=int, default=4)  # convolution kernel size


def selective_scan(x, Δ, A, B, C, h_0):
    
    def _scan(h_tm1, inputs):
        # inputs are slices of x, Δ, B, C on leading axis
        x_t, _Δ, _B, _C = inputs
        # Equation 4 in paper
        ΔA = _Δ * A
        A_bar = jnp.exp(ΔA)
        ΔB = _Δ * _B
        # Identity matrix with shape [B, dim_h, dim_h]
        I = jnp.eye(A.shape[1])
        I = jnp.broadcast_to(I[None, :, :], (A.shape[0], I.shape[0], I.shape[1]))
        # Equation 4 in paper
        B_bar = jnp.linalg.inv(ΔA)*(A_bar - I)*ΔB
        # Equations 2a from paper
        h_t = jnp.einsum('bij,bj->bi', A_bar, h_tm1)
        h_t += jnp.einsum('bij,bj->bi', B_bar, x_t)
        # Equation 2b from paper
        y_t = jnp.einsum('bij,bj->bj', _C, h_t)
        return h_t, y_t

    # Broadcast A to shape [B, dim_h, dim_h]
    A = jnp.broadcast_to(A[None, :, :], (x.shape[0], A.shape[0], A.shape[1]))
    # reshape x to [B, L, dim_h]
    x = jnp.moveaxis(x, 1, 0)
    # reshape Δ, B, C from [B, L, dim_h] to [L, B, 1, dim_h]
    Δ = jnp.moveaxis(Δ, 1, 0)[:, :, None, :]
    B = jnp.moveaxis(B, 1, 0)[:, :, None, :]
    C = jnp.moveaxis(C, 1, 0)[:, :, None, :]
    # forward scan
    _, out1 = lax.scan(_scan, h_0, (x, Δ, B, C))
    # backward scan
    _, out2 = lax.scan(_scan, h_0, (x, Δ, B, C), reverse=True)
    # merge forward and backward scan outputs
    out = out1 + out2
    # reshape outputs back to [B, L, dim_h]
    out = jnp.moveaxis(out, 0, 1)
    return out


def mamba_block(x, params):
    # input sequence x with shape [B, L, dim_c]
    # B is batch size
    # L is sequence length
    # project input x to hidden dimmension
    # (B, L, dim_c) @ (dim_c, dim_h) -> (B, L, dim_h)
    x = x @ params["in_proj_w"] + params["in_proj_b"]
    # skip connection comes out
    x_skip = jax.nn.silu(x)
    # project input sequence x to B (input matrix)
    # (B, L, dim_h) @ (dim_h, dim_h) -> (B, L, dim_h)
    B = x @ params["B_proj_w"]
    # project input sequence x to C (output matrix)
    # (B, L, dim_h) @ (dim_h, dim_h) -> (B, L, dim_h)
    C = x @ params["C_proj_w"]
    # project input x to Δ
    # (B, L, dim_h) @ (dim_h, dim_Δ) -> (B, L, dim_Δ)
    Δ = x @ params["Δ_proj_w"]
    # broadcast Δ to shape (B, L, dim_h)
    # Δ = jnp.broadcast_to(Δ[:, :, None], (Δ.shape[0], Δ.shape[1], params["A"].shape[1]))
    # causal 1D convolution layer
    x = jax.lax.conv_general_dilated(
        x,  # input sequence x with shape [B, L, dim_h]
        params["conv_w"],  # convolution kernel with shape [dim_conv, dim_h, dim_h]
        window_strides=(1,),  # stride of 1
        padding=[(params["conv_w"].shape[0] - 1, 0)],  # causal padding
        lhs_dilation=(1,),
        rhs_dilation=(1,),
        dimension_numbers=("NWC", "WIO", "NWC"),
    )
    x = jax.nn.silu(x)
    # initial hidden state h [B, dim_h]
    h_0 = jnp.broadcast_to(params["h_0"], (x.shape[0], params["h_0"].shape[0]))
    x = selective_scan(x, Δ, params["A"], B, C, h_0)
    # skip connection goes back in
    # (B, L, dim_h) * (B, L, dim_h) -> (B, L, dim_h)
    x = x * x_skip
    # project merged x to output y
    # (B, L, dim_h) @ (dim_h, dim_c) -> (B, L, dim_c)
    y = x @ params["out_proj_w"] + params["out_proj_b"]
    # output sequence y with shape [B, L, dim_c]
    return y


def rms_norm(x, weight, bias, eps=1e-6):
    variance = jnp.mean(x**2, axis=-1, keepdims=True)
    normalized_x = x * jax.lax.rsqrt(variance + eps)
    return normalized_x * weight + bias


def model(x, params):
    # model is a stack of mamba blocks
    for block_params in params["residual_blocks"]:
        y = mamba_block(x, block_params["mamba_params"])
        # apply rms norm after each block
        y = rms_norm(y, block_params["norm_w"], block_params["norm_b"])
        # skip connection
        x += y
    # remove channel dimmension
    x = jnp.squeeze(x, axis=-1)
    # classification head
    logits = x @ params["class_head_w"] + params["class_head_b"]
    return logits


def cross_entropy_loss(params, batch):
    """Computes cross-entropy loss for Mamba model on MNIST."""
    images, labels = batch
    logits = model(images, params)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * labels, axis=-1))


def accuracy(params, batch):
    """Computes accuracy for Mamba model on MNIST."""
    images, labels = batch
    predicted_labels = jnp.argmax(model(images, params), axis=-1)
    return jnp.mean(predicted_labels == jnp.argmax(labels, axis=-1))


if __name__ == "__main__":
    args = parser.parse_args()
    check_gpu()
    rng = random.key(args.seed)

    # Load MNIST dataset
    train_images, train_labels, test_images, test_labels = mnist(data_dir=args.data_dir)
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, args.batch_size)
    num_batches = num_complete_batches + bool(leftover)
    num_classes = 10
    dim_c = 1
    dim_seq = 28 * 28

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * args.batch_size : (i + 1) * args.batch_size]
                yield jnp.expand_dims(train_images[batch_idx], axis=-1), train_labels[
                    batch_idx
                ]

    batches = data_stream()

    params = {
        "residual_blocks": [
            {
                "mamba_params": {
                    # input projection
                    "in_proj_w": random.normal(rng, (dim_c, args.dim_h)),
                    "in_proj_b": jnp.zeros(args.dim_h),
                    # SSM parameters
                    "B_proj_w": random.normal(rng, (args.dim_h, args.dim_h)),
                    "C_proj_w": random.normal(rng, (args.dim_h, args.dim_h)),
                    # Initialization for Δ from paper in Section 3.6
                    "Δ_proj_w": random.uniform(
                        rng, (args.dim_h, args.dim_Δ), minval=0.001, maxval=0.1
                    ),
                    # Initialization for A from paper in Section 3.6 and Table 8
                    "A": jnp.eye(args.dim_h)
                    + jnp.tril(jnp.ones((args.dim_h, args.dim_h)), -1),
                    # causal 1D convolution layer
                    "conv_w": random.normal(
                        rng, (args.dim_conv, args.dim_h, args.dim_h)
                    ),
                    # initial hidden state h_0
                    "h_0": random.normal(rng, (args.dim_h,)),
                    # output projection
                    "out_proj_w": random.normal(rng, (args.dim_h, dim_c)),
                    "out_proj_b": jnp.zeros(dim_c),
                },
                # RMS normalization layer
                "norm_w": jnp.ones(dim_c),
                "norm_b": jnp.zeros(dim_c),
            }
            for _ in range(args.num_blocks)
        ],
        # classification head
        "class_head_w": random.normal(rng, (dim_seq, num_classes)),
        "class_head_b": jnp.zeros(num_classes),
    }

    # Optimizer
    opt_init, opt_update, get_params = optimizers.adam(
        args.learning_rate, args.b1, args.b2
    )

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(cross_entropy_loss)(params, batch), opt_state)

    opt_state = opt_init(params)
    itercount = itertools.count()

    print("\nStarting training...")
    for epoch in range(args.num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
            opt_state = update(next(itercount), opt_state, next(batches))
        epoch_time = time.time() - start_time

        params = get_params(opt_state)
        train_acc = accuracy(params, (train_images, train_labels))
        test_acc = accuracy(params, (test_images, test_labels))
        print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
        print(f"Training set accuracy {train_acc}")
        print(f"Test set accuracy {test_acc}")
