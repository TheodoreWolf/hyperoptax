import jax
import jax.tree_util as pytree


class Domain(pytree.PyTree):
    def make_meshgrid(self, domain: jax.Array):
        pass
