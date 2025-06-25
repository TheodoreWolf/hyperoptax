import jax
import jax.tree_util as pytree


class Domain(pytree.PyTree):
    pass
    def make_meshgrid(self, domain: jax.Array):
