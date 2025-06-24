import jax

class Domain:
    def __init__(self, domain: jax.Array):
        self.domain = domain

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.domain[x]

    def __len__(self) -> int:
        return len(self.domain)
