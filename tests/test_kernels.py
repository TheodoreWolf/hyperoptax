import jax
import jax.numpy as jnp

from hyperoptax.kernels import RBF, Matern


class TestRBF:
    def setup_method(self):
        self.x = jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        self.y = jnp.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])

    def test_rbf_with_same_points(self):
        kernel = RBF(length_scale=1.0)
        # correlation matrix should be all ones
        assert jnp.allclose(kernel(self.x, self.x), kernel.diag(self.x))
        assert jnp.allclose(kernel(self.x, self.x), jnp.full((3, 3), 1.0))

    def test_rbf_with_different_points(self):
        kernel = RBF(length_scale=1.0)
        # correlation matrix should be all exp(-1)
        assert jnp.allclose(kernel(self.x, self.y), jnp.full((3, 3), jnp.exp(-1)))

    def test_rbf_with_different_data_sizes(self):
        kernel = RBF(length_scale=1.0)
        x = jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        # correlation matrix should be all exp(-1) with shape (4, 3)
        assert jnp.allclose(kernel(x, self.y), jnp.full((4, 3), jnp.exp(-1)))

    def test_rbf_with_different_length_scales(self):
        kernel = RBF(length_scale=2.0)
        # correlation matrix should be all exp(-1/4) with shape (3, 3)
        assert jnp.allclose(kernel(self.x, self.y), jnp.full((3, 3), jnp.exp(-1 / 4)))


class TestMatern:
    def setup_method(self):
        self.x = jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        self.y = jnp.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])

    def test_matern_with_same_points(self):
        kernel = Matern(length_scale=1.0)
        # correlation matrix should be all ones
        assert jnp.allclose(kernel(self.x, self.x), jnp.full((3, 3), 1.0))

    def test_matern_with_different_points(self):
        kernel = Matern(length_scale=1.0, nu=0.5)
        # correlation matrix should be all exp(-sqrt(2))
        assert jnp.allclose(
            kernel(self.x, self.y), jnp.full((3, 3), jnp.exp(-jnp.sqrt(2)))
        )

    def test_matern_with_jit(self):
        kernel = Matern(length_scale=1.0, nu=0.5)
        # correlation matrix should be all exp(-sqrt(2))
        jitted_kernel = jax.jit(kernel)
        assert jnp.allclose(
            jitted_kernel(self.x, self.y), jnp.full((3, 3), jnp.exp(-jnp.sqrt(2)))
        )
