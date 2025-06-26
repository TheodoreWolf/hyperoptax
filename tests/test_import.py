"""Test that the package can be imported correctly."""


def test_import_hyperoptax():
    """Test that hyperoptax can be imported."""
    import hyperoptax

    assert hyperoptax is not None


def test_import_modules():
    """Test that all modules can be imported."""
    from hyperoptax import grid_search, spaces, base

    assert grid_search is not None
    assert spaces is not None
    assert base is not None
