import jax
import jax.numpy as jnp
import pytest

from hyperoptax.kernels import RBF, Matern, BaseKernel

x = jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
y = jnp.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])
x_long = jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])


class TestRBF:
    def test_rbf_with_same_points(self):
        kernel = RBF(length_scale=1.0)
        # correlation matrix should be all ones
        assert jnp.allclose(kernel(x, x), jnp.full((3, 3), 1.0))

    def test_rbf_with_different_points(self):
        kernel = RBF(length_scale=1.0)
        # correlation matrix should be all exp(-1)
        assert jnp.allclose(kernel(x, y), jnp.full((3, 3), jnp.exp(-1)))

    def test_rbf_with_different_data_sizes(self):
        kernel = RBF(length_scale=1.0)
        # correlation matrix should be all exp(-1) with shape (4, 3)
        assert jnp.allclose(kernel(x_long, y), jnp.full((4, 3), jnp.exp(-1)))

    def test_rbf_with_different_length_scales(self):
        kernel = RBF(length_scale=2.0)
        # correlation matrix should be all exp(-1/4) with shape (3, 3)
        assert jnp.allclose(kernel(x, y), jnp.full((3, 3), jnp.exp(-1 / 4)))


class TestMatern:
    def test_matern_with_same_points(self):
        kernel = Matern(length_scale=1.0)
        # correlation matrix should be all ones
        assert jnp.allclose(kernel(x, x), jnp.full((3, 3), 1.0))

    def test_matern_with_different_points(self):
        kernel = Matern(length_scale=1.0, nu=0.5)
        # correlation matrix should be all exp(-sqrt(2))
        assert jnp.allclose(kernel(x, y), jnp.full((3, 3), jnp.exp(-jnp.sqrt(2))))

    def test_matern_with_jit(self):
        kernel = Matern(length_scale=1.0, nu=0.5)
        # correlation matrix should be all exp(-sqrt(2))
        jitted_kernel = jax.jit(kernel)
        assert jnp.allclose(
            jitted_kernel(x, y), jnp.full((3, 3), jnp.exp(-jnp.sqrt(2)))
        )

    def test_matern_nu_1_5(self):
        kernel = Matern(length_scale=1.0, nu=1.5)
        d = jnp.sqrt(2.0)  # distance between x[0] and y[0]
        K = jnp.sqrt(3) * d
        expected = (1 + K) * jnp.exp(-K)
        assert jnp.allclose(kernel(x, y), jnp.full((3, 3), expected), atol=1e-6)

    def test_matern_nu_2_5(self):
        kernel = Matern(length_scale=1.0, nu=2.5)
        d = jnp.sqrt(2.0)
        K = jnp.sqrt(5) * d
        expected = (1 + K + K**2 / 3) * jnp.exp(-K)
        assert jnp.allclose(kernel(x, y), jnp.full((3, 3), expected), atol=1e-6)

    def test_matern_nu_inf(self):
        # nu=inf should be equivalent to the RBF (squared-exponential) kernel
        matern_inf = Matern(length_scale=1.0, nu=jnp.inf)
        rbf = RBF(length_scale=1.0)
        assert jnp.allclose(matern_inf(x, y), rbf(x, y), atol=1e-6)

    def test_matern_invalid_nu_raises(self):
        kernel = Matern(length_scale=1.0, nu=3.0)
        with pytest.raises(ValueError, match="not supported"):
            kernel(x, y)


class TestBaseKernel:
    def test_base_kernel_call_raises(self):
        class MinimalKernel(BaseKernel):
            def __call__(self, x, y, length_scale=None):
                return super().__call__(x, y, length_scale)

        with pytest.raises(NotImplementedError):
            MinimalKernel()(x, y)

