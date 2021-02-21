"""
code is from deep mind research git repository
https://github.com/deepmind/deepmind-research/tree/master/nfnets

I have just taken out main changes mentioned in paper from code and added my comment
on top of each part for rapid understanding. all code is from original code published with paper by deepmind.
"""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


############################ Part 1 ########################################
# activations are applied after convolution operation.
# key note here is "Magic constant" multiplied to each activation.

nonlinearities = {
    'identity': lambda x: x,
    'celu': lambda x: jax.nn.celu(x) * 1.270926833152771,
    'elu': lambda x: jax.nn.elu(x) * 1.2716004848480225,
    'gelu': lambda x: jax.nn.gelu(x) * 1.7015043497085571,
    'glu': lambda x: jax.nn.glu(x) * 1.8484294414520264,
    'leaky_relu': lambda x: jax.nn.leaky_relu(x) * 1.70590341091156,
    'log_sigmoid': lambda x: jax.nn.log_sigmoid(x) * 1.9193484783172607,
    'log_softmax': lambda x: jax.nn.log_softmax(x) * 1.0002083778381348,
    'relu': lambda x: jax.nn.relu(x) * 1.7139588594436646,
    'relu6': lambda x: jax.nn.relu6(x) * 1.7131484746932983,
    'selu': lambda x: jax.nn.selu(x) * 1.0008515119552612,
    'sigmoid': lambda x: jax.nn.sigmoid(x) * 4.803835391998291,
    'silu': lambda x: jax.nn.silu(x) * 1.7881293296813965,
    'soft_sign': lambda x: jax.nn.soft_sign(x) * 2.338853120803833,
    'softplus': lambda x: jax.nn.softplus(x) * 1.9203323125839233,
    'tanh': lambda x: jnp.tanh(x) * 1.5939117670059204,
}


############################ Part 2 ########################################


# instead of regular convolutions, weights standardized convolution is implemented to
# capture local relations.

class WSConv2D(hk.Conv2D):
    """2D Convolution with Scaled Weight Standardization and affine gain+bias."""

    @hk.transparent
    def standardize_weight(self, weight, eps=1e-4):
        """Apply scaled WS with affine gain."""
        mean = jnp.mean(weight, axis=(0, 1, 2), keepdims=True)
        var = jnp.var(weight, axis=(0, 1, 2), keepdims=True)
        fan_in = np.prod(weight.shape[:-1])
        # Get gain
        gain = hk.get_parameter('gain', shape=(weight.shape[-1],),
                                dtype=weight.dtype, init=jnp.ones)
        # Manually fused normalization, eq. to (w - mean) * gain / sqrt(N * var)
        scale = jax.lax.rsqrt(jnp.maximum(var * fan_in, eps)) * gain
        shift = mean * scale
        return weight * scale - shift

    def __call__(self, inputs: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
        w_shape = self.kernel_shape + (
            inputs.shape[self.channel_index] // self.feature_group_count,
            self.output_channels)
        # Use fan-in scaled init, but WS is largely insensitive to this choice.
        w_init = hk.initializers.VarianceScaling(1.0, 'fan_in', 'normal')
        w = hk.get_parameter('w', w_shape, inputs.dtype, init=w_init)
        weight = self.standardize_weight(w, eps)
        out = jax.lax.conv_general_dilated(
            inputs, weight, window_strides=self.stride, padding=self.padding,
            lhs_dilation=self.lhs_dilation, rhs_dilation=self.kernel_dilation,
            dimension_numbers=self.dimension_numbers,
            feature_group_count=self.feature_group_count)
        # Always add bias
        bias_shape = (self.output_channels,)
        bias = hk.get_parameter('bias', bias_shape, inputs.dtype, init=jnp.zeros)
        return out + bias

############################ Part 3 ########################################

# stochastic dropout applied in each block.
class StochDepth(hk.Module):
    """Batchwise Dropout used in EfficientNet, optionally sans rescaling."""

    def __init__(self, drop_rate, scale_by_keep=False, name=None):
        super().__init__(name=name)
        self.drop_rate = drop_rate
        self.scale_by_keep = scale_by_keep

    def __call__(self, x, is_training) -> jnp.ndarray:
        if not is_training:
            return x
        batch_size = x.shape[0]
        r = jax.random.uniform(hk.next_rng_key(), [batch_size, 1, 1, 1],
                               dtype=x.dtype)
        keep_prob = 1. - self.drop_rate
        binary_tensor = jnp.floor(keep_prob + r)
        if self.scale_by_keep:
            x = x / keep_prob
        return x * binary_tensor


############################ Part 4 ########################################

# Adaptive Gradient clipping in optimizer
class SGD_AGC(Optimizer):  # pylint:disable=invalid-name
  """SGD with Unit-Adaptive Gradient-Clipping.

  References:
    [Brock, Smith, De, Simonyan 2021] High-Performance Large-Scale Image
    Recognition Without Normalization.
  """
  defaults = {'weight_decay': None, 'momentum': None, 'dampening': 0,
              'nesterov': None, 'clipping': 0.01, 'eps': 1e-3}

  def __init__(self, params, lr, weight_decay=None,
               momentum=None, dampening=0, nesterov=None,
               clipping=0.01, eps=1e-3):
    super().__init__(
        params, defaults={'lr': lr, 'weight_decay': weight_decay,
                          'momentum': momentum, 'dampening': dampening,
                          'clipping': clipping, 'nesterov': nesterov,
                          'eps': eps})

  def create_buffers(self, name, param):
    return SGD.create_buffers(self, name, param)

  def update_param(self, param, grad, state, opt_params):
    """Clips grads if necessary, then applies the optimizer update."""
    if param is None:
      return param, state

    ### Gradient clipping logic
    if opt_params['clipping'] is not None:
      param_norm = jnp.maximum(unitwise_norm(param), opt_params['eps'])
      grad_norm = unitwise_norm(grad)
      max_norm = param_norm * opt_params['clipping']
      # If grad norm > clipping * param_norm, rescale
      trigger = grad_norm > max_norm
      # Note the max(||G||, 1e-6) is technically unnecessary here, as
      # the clipping shouldn't trigger if the grad norm is zero,
      # but we include it in practice as a "just-in-case".
      clipped_grad = grad * (max_norm / jnp.maximum(grad_norm, 1e-6))
      grad = jnp.where(trigger, clipped_grad, grad)
    return SGD.update_param(self, param, grad, state, opt_params)

