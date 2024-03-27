import jax
import jax.numpy as jnp
from jax import random, grad, jit, lax
from jax.example_libraries import optimizers
from jax.examples import datasets

# Define helper functions for key components

def linear(x, weight, bias):
  """Applies a linear transformation to the input."""
  return x @ weight + bias

def causal_conv1d(x, weight, bias):
  """Performs causal convolution on the input sequence."""
  padding = [(weight.shape[-1] - 1, 0)]  # Causal padding
  out = lax.conv(x, weight, (1,), padding, dimension_numbers=('NWC', 'WIO', 'NWC'))  # Use JAX's built-in padding
  return out + bias

def selective_scan(input_sequence, delta, state_transition_matrix, input_matrix, output_matrix, skip_matrix):
  """Implements the selective scan algorithm using jax.lax.scan."""
  def scan_func(state, inputs):
    current_input, delta, input_matrix, output_matrix = inputs
    updated_state = jnp.exp(delta @ state_transition_matrix) * state + delta @ input_matrix * current_input
    output = updated_state @ output_matrix
    return updated_state, output
  
  initial_state = jnp.zeros_like(input_sequence[:, :, :1])
  _, outputs = lax.scan(scan_func, initial_state, (input_sequence, delta, input_matrix, output_matrix))
  return outputs + input_sequence * skip_matrix

def rms_norm(x, weight, bias, eps=1e-6):
  """Applies RMS normalization to the input."""
  variance = jnp.mean(x**2, axis=-1, keepdims=True)
  normalized_x = x * jax.lax.rsqrt(variance + eps)
  return normalized_x * weight + bias


# MambaBlock function

def mamba_block(x, params):
  """Implements a single Mamba block."""
  # Unpack parameters
  input_proj_weight, input_proj_bias, conv_weight, conv_bias, x_proj_weight, dt_proj_weight, dt_proj_bias, state_transition_matrix_log, skip_matrix, output_proj_weight, output_proj_bias = params

  # Project input
  projected_input = linear(x, input_proj_weight, input_proj_bias)
  x, z = jnp.split(projected_input, 2, axis=-1)

  # Causal convolution
  x = causal_conv1d(x, conv_weight, conv_bias)
  x = jax.nn.silu(x)

  # Compute delta, input matrix, and output matrix
  x_dbl = linear(x, x_proj_weight, None)
  dt_rank, d_state = dt_proj_weight.shape[0], state_transition_matrix_log.shape[1]
  delta, input_matrix, output_matrix = jnp.split(x_dbl, [dt_rank, d_state, d_state], axis=-1)
  delta = linear(delta, dt_proj_weight, dt_proj_bias)
  delta = jax.nn.softplus(delta)

  # Selective scan
  state_transition_matrix = -jnp.exp(state_transition_matrix_log)
  y = selective_scan(x, delta, state_transition_matrix, input_matrix, output_matrix, skip_matrix)

  # Output projection
  out = linear(y, output_proj_weight, output_proj_bias)
  return out

# ResidualBlock function

def residual_block(x, params):
  """Implements a ResidualBlock with MambaBlock and RMS normalization."""
  # Unpack parameters
  mamba_params, norm_weight, norm_bias = params

  # Apply MambaBlock
  y = mamba_block(x, mamba_params)

  # Apply RMS normalization
  y = rms_norm(y, norm_weight, norm_bias)

  # Add residual connection
  out = x + y
  return out

# Loss function and accuracy metric

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

# Model construction and training loop

def model(images, params):
  """Mamba model for MNIST classification."""
  # Flatten images
  flat_images = jnp.reshape(images, (images.shape[0], -1))

  # Apply ResidualBlocks
  x = flat_images
  for block_params in params["residual_blocks"]:
    x = residual_block(x, block_params)

  # Final linear layer for classification
  logits = linear(x, params["output_weight"], params["output_bias"])
  return logits

if __name__ == "__main__":
  # Set random seed
  rng = random.PRNGKey(0)

  # Hyperparameters (adjust as needed)
  step_size = 0.001
  num_epochs = 10
  batch_size = 128
  momentum_mass = 0.9

  # Load MNIST dataset
  train_images, train_labels, test_images, test_labels = datasets.mnist()
  num_train = train_images.shape[0]

  # Estimated parameter values (adjust based on your model configuration)
  d_model = 2560  # Example hidden dimension
  d_state = 16  # Example state dimension
  d_conv = 4  # Example convolution kernel size
  expand = 2  # Example expansion factor
  dt_rank = 160  # Example delta rank
  dt_min = 0.001  # Example minimum delta value
  dt_max = 0.1  # Example maximum delta value
  vocab_size = 10  # MNIST has 10 classes

  # Initialize dt bias based on dt_min and dt_max
  dt_init_std = dt_rank**-0.5
  dt_init_floor = 1e-4
  dt = jnp.exp(random.uniform(jax.random.PRNGKey(6), (expand * d_model,), minval=jnp.log(dt_min), maxval=jnp.log(dt_max))).clip(min=dt_init_floor)
  inv_dt = dt + jnp.log(-jnp.expm1(-dt))

  # Create parameter dictionary
  params = {
      "residual_blocks": [
          # Example parameters for one ResidualBlock (repeat for multiple blocks)
          {
              "mamba_params": [
                  random.normal(rng, (d_model, 2 * expand * d_model)),
                  jnp.zeros(2 * expand * d_model),
                  random.normal(rng, (expand * d_model, 1, d_conv)),
                  jnp.zeros(expand * d_model),
                  random.normal(rng, (expand * d_model, dt_rank + 2 * d_state)),
                  random.normal(rng, (dt_rank, expand * d_model)),
                  inv_dt,  # Initialized based on dt_min/dt_max
                  jnp.log(jnp.arange(1, d_state + 1, dtype=jnp.float32)),
                  jnp.ones(expand * d_model),
                  random.normal(rng, (expand * d_model, d_model)),
                  jnp.zeros(d_model),
              ],
              "norm_weight": jnp.ones(d_model),
              "norm_bias": jnp.zeros(d_model),
          },
          # ... (Add parameters for more ResidualBlocks) ...
      ],
      "output_weight": random.normal(rng, (d_model, 10)),
      "output_bias": jnp.zeros(10),
  }

  # Data stream function
  def data_stream():
    rng = random.PRNGKey(0)
    while True:
      perm = random.permutation(rng, num_train)
      for i in range(num_train // batch_size):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]

  # Optimizer setup
  opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)

  # Update function
  @jit
  def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(cross_entropy_loss)(params, batch), opt_state)

  # Initialize optimizer state
  opt_state = opt_init(params)

  # Training loop
  for epoch in range(num_epochs):
    for batch in data_stream():
      opt_state = update(0, opt_state, batch)  # Assuming no gradient accumulation

    # Evaluate accuracy on train and test sets
    train_acc = accuracy(get_params(opt_state), (train_images, train_labels))
    test_acc = accuracy(get_params(opt_state), (test_images, test_labels))
    print(f"Epoch {epoch} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")