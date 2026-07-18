"""Stable, human-readable names for dashboard trials."""

from __future__ import annotations

from hashlib import blake2s

# These v1 word lists are part of the display-name contract. Keep their order
# stable so names assigned to historical trials do not change between releases.
_ADJECTIVES = (
    "amber",
    "brisk",
    "calm",
    "clever",
    "coral",
    "cosmic",
    "crisp",
    "daring",
    "eager",
    "electric",
    "gentle",
    "golden",
    "hidden",
    "ivory",
    "jolly",
    "lucid",
    "lunar",
    "mellow",
    "misty",
    "nimble",
    "noble",
    "quiet",
    "rapid",
    "silver",
    "solar",
    "steady",
    "sunny",
    "swift",
    "vivid",
    "warm",
    "wild",
    "zen",
)

_NOUNS = (
    "badger",
    "cedar",
    "comet",
    "dolphin",
    "falcon",
    "fern",
    "fox",
    "glacier",
    "heron",
    "iris",
    "juniper",
    "koala",
    "lark",
    "lynx",
    "maple",
    "meadow",
    "otter",
    "panda",
    "pebble",
    "pine",
    "raven",
    "reef",
    "river",
    "robin",
    "sparrow",
    "spruce",
    "star",
    "tiger",
    "valley",
    "wave",
    "willow",
    "wolf",
)


def make_trial_name(
    trial_id: str,
    *,
    evaluation_index: int | None,
    batch_index: int,
) -> str:
    """Return a deterministic ``adjective-noun-number`` display name.

    The human-facing number is one-based. Trials without an evaluation index
    use their batch number; the digest-derived word pair still distinguishes
    parallel slots.
    """

    if not trial_id:
        raise ValueError("trial_id must not be empty")
    index = evaluation_index if evaluation_index is not None else batch_index
    if index < 0:
        raise ValueError("trial number source must not be negative")

    digest = blake2s(
        trial_id.encode("utf-8"),
        digest_size=4,
        person=b"hypx-v1",
    ).digest()
    adjective = _ADJECTIVES[int.from_bytes(digest[:2], "big") % len(_ADJECTIVES)]
    noun = _NOUNS[int.from_bytes(digest[2:], "big") % len(_NOUNS)]
    return f"{adjective}-{noun}-{index + 1}"


__all__ = ["make_trial_name"]
