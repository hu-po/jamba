import jax

def check_gpu():
  print("jax.devices:", jax.devices())
  print("jax.default_backend():", jax.default_backend())
  try:
      _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
      return True
  except:
      return False