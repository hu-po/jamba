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
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default="test1")

parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--momentum", type=float, default=0.9)

parser.add_argument("--data_dir", type=str, default="/home/oop/dev/data/mnist")
parser.add_argument("--ckpt_dir", type=str, default="/ckpt")
parser.add_argument("--save_ckpt", type=bool, default=False)
parser.add_argument("--logs_dir", type=str, default="/logs")

# Mamba model hyperparameters
parser.add_argument("--num_blocks", type=int, default=1)  # number of mamba blocks
parser.add_argument("--mod_dim_D", type=int, default=64)  # Model Dimension D
parser.add_argument("--exp_fac_E", type=int, default=2)  # Expansion Factor E

parser.add_argument("--d_state", type=int, default=16)  # state dimension
parser.add_argument("--d_conv", type=int, default=4)  # convolution kernel size
parser.add_argument("--dt_rank", type=int, default=160)  # delta rank
parser.add_argument("--dt_min", type=int, default=0.001)  # minimum delta value
parser.add_argument("--dt_max", type=int, default=0.1)  # maximum delta value


def selective_scan(
    input_sequence,
    delta,
    state_transition_matrix,
    input_matrix,
    output_matrix,
    skip_matrix,
):

    def scan_func(state, inputs):
        current_input, delta, input_matrix, output_matrix = inputs
        updated_state = (
            jnp.exp(delta @ state_transition_matrix) * state
            + delta @ input_matrix * current_input
        )
        output = updated_state @ output_matrix
        return updated_state, output

    initial_state = jnp.zeros_like(input_sequence[:, :, :1])
    _, outputs = lax.scan(
        scan_func, initial_state, (input_sequence, delta, input_matrix, output_matrix)
    )
    return outputs + input_sequence * skip_matrix


def mamba_block(x, params):
    # input sequence x with shape [B, L, D]
    # B is batch size, L is sequence length, D is token dimension

    # project input x to Δ, B, C
    x = x @ params['in_proj_w'] + params['in_proj_b']

    # split projected input x into two branches
    x1, x2 = jnp.split(x, 2, axis=-1)

    # branch 1 goes to conv, silu, scan
    x1 = jax.lax.conv_general_dilated(
        x1,
        params['conv_w'],
        window_strides=(1,),
        # Padding for causality: only pad the left side of the sequence
        # (left_pad, right_pad) for each spatial dimension
        padding = [(params['conv_w'].shape[0] - 1, 0)],
        lhs_dilation=(1,),
        rhs_dilation=(1,),
        dimension_numbers=('NWC', 'WIO', 'NWC'),  # specify data format: batch, spatial, channel
        feature_group_count=1,  # for grouped (channel-wise) convolution, set >1
    ) + params['conv_b']
    x1 = jax.nn.silu(x1)
    x1 = selective_scan()

    # branch2 goes to silu
    x2 = jax.nn.silu(x2)

    # merge branch 1 and branch 2
    x = x1 * x2

    # project merged x to output y
    y = x @ params['out_proj_w'] + params['out_proj_b']

    # output sequence y with shape [B, L, D]
    return y

def rms_norm(x, weight, bias, eps=1e-6):
    variance = jnp.mean(x**2, axis=-1, keepdims=True)
    normalized_x = x * jax.lax.rsqrt(variance + eps)
    return normalized_x * weight + bias


def model(x, params):
    # model is a stack of mamba blocks
    for block_params in params["residual_blocks"]:
        y = mamba_block(x, block_params['mamba_params'])
        # apply rms norm after each block
        y = rms_norm(y, block_params['norm_w'], block_params['norm_b'])
        # skip connection
        x += y
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

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * args.batch_size : (i + 1) * args.batch_size]
                yield jnp.expand_dims(train_images[batch_idx], axis=-1), train_labels[batch_idx]

    batches = data_stream()

    # Parameters
    dt_init_std = args.dt_rank**-0.5
    dt_init_floor = 1e-4
    dt = jnp.exp(
        random.uniform(
            jax.random.PRNGKey(6),
            (args.exp_fac_E * args.mod_dim_D,),
            minval=jnp.log(args.dt_min),
            maxval=jnp.log(args.dt_max),
        )
    ).clip(min=dt_init_floor)
    inv_dt = dt + jnp.log(-jnp.expm1(-dt))
    params = {
        "residual_blocks": [
            {
                "mamba_params": {
                    # input linear projection layer
                    "in_proj_w" : random.normal(rng, (args.mod_dim_D, 2 * args.exp_fac_E * args.mod_dim_D)),
                    "in_proj_b" : jnp.zeros(2 * args.exp_fac_E * args.mod_dim_D),

                    # causal 1D convolution layer
                    "conv_w" : random.normal(rng, (args.d_conv, args.exp_fac_E * args.mod_dim_D, 1)),
                    "conv_b" : jnp.zeros(1),

                    "x_proj_weight" : random.normal(rng, (args.exp_fac_E * args.mod_dim_D, args.dt_rank + 2 * args.d_state)),  # x_proj_weight
                    "dt_proj_weight" : random.normal(rng, (args.exp_fac_E * args.mod_dim_D, args.dt_rank)),  # dt_proj_weight
                    "dt_proj_bias" : jnp.log(dt),  # dt_proj_bias (initialized based on dt_min/dt_max)
                    "state_transition_matrix_log" : random.normal(rng, (args.d_state, args.d_state)),  # state_transition_matrix_log
                    "skip_matrix" : jnp.ones(args.exp_fac_E * args.mod_dim_D),  # skip_matrix
                    
                    # output linear projection layer
                    "out_proj_w" : random.normal(rng, (args.d_state, args.exp_fac_E * args.mod_dim_D)),
                    "out_proj_b" : jnp.zeros(args.exp_fac_E * args.mod_dim_D),
                },
                # RMS normalization layer
                "norm_w": jnp.ones(args.exp_fac_E * args.mod_dim_D),
                "norm_b": jnp.zeros(args.exp_fac_E * args.mod_dim_D),
            }
            for _ in range(args.num_blocks)
        ],
        # classification head
        "class_head_w": random.normal(rng, (args.exp_fac_E * args.mod_dim_D, 10)),
        "class_head_b": jnp.zeros(10),
    }

    # Optimizer
    opt_init, opt_update, get_params = optimizers.momentum(
        args.learning_rate, mass=args.momentum
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
