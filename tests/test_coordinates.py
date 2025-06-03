from catley.util.coordinates import CoordinateConverter


def test_pixel_to_tile_basic():
    conv = CoordinateConverter(10, 10, 16, 16, 160, 160)
    assert conv.pixel_to_tile(0, 0) == (0, 0)
    assert conv.pixel_to_tile(159, 159) == (9, 9)
    assert conv.pixel_to_tile(160, 160) == (9, 9)


def test_pixel_to_tile_scaled():
    conv = CoordinateConverter(10, 10, 16, 16, 80, 80)
    # scale factors 2
    assert conv.pixel_to_tile(20, 20) == (2, 2)
    assert conv.pixel_to_tile(-5, -5) == (0, 0)
    assert conv.pixel_to_tile(1000, 1000) == (9, 9)
