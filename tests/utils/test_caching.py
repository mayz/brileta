import pytest

from brileta.util.caching import CacheStats, ResourceCache


class TestCacheStats:
    def test_initial_state(self) -> None:
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.total_lookups == 0
        assert stats.hit_rate == 0.0

    def test_total_lookups_calculation(self) -> None:
        stats = CacheStats(hits=5, misses=3)
        assert stats.total_lookups == 8

    def test_hit_rate_calculation(self) -> None:
        stats = CacheStats(hits=7, misses=3)
        assert stats.hit_rate == 70.0

    def test_hit_rate_with_zero_lookups(self) -> None:
        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0

    def test_repr(self) -> None:
        stats = CacheStats(hits=12, misses=8)
        expected = "12 hits, 8 misses (60.0% hit rate)"
        assert repr(stats) == expected


class TestResourceCache:
    def test_init_with_defaults(self) -> None:
        cache: ResourceCache[str, int] = ResourceCache("test")
        assert cache.name == "test"
        assert cache.max_size == 16
        assert len(cache) == 0
        assert cache.on_evict is None
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0

    def test_init_with_custom_params(self) -> None:
        def evict_callback(x: int) -> None:
            pass

        cache: ResourceCache[str, int] = ResourceCache(
            "custom", max_size=5, on_evict=evict_callback
        )
        assert cache.name == "custom"
        assert cache.max_size == 5
        assert cache.on_evict is evict_callback

    def test_init_with_invalid_max_size(self) -> None:
        with pytest.raises(
            ValueError, match="Cache max_size must be a positive integer"
        ):
            ResourceCache("test", max_size=0)

        with pytest.raises(
            ValueError, match="Cache max_size must be a positive integer"
        ):
            ResourceCache("test", max_size=-1)

    def test_store_and_get_basic(self) -> None:
        cache: ResourceCache[str, int] = ResourceCache("test")

        # Store an item
        cache.store("key1", 42)
        assert len(cache) == 1

        # Retrieve the item
        result = cache.get("key1")
        assert result == 42
        assert cache.stats.hits == 1
        assert cache.stats.misses == 0

    def test_get_nonexistent_key(self) -> None:
        cache: ResourceCache[str, int] = ResourceCache("test")

        result = cache.get("nonexistent")
        assert result is None
        assert cache.stats.hits == 0
        assert cache.stats.misses == 1

    def test_lru_behavior(self) -> None:
        cache: ResourceCache[str, int] = ResourceCache("test", max_size=2)

        # Fill cache to capacity
        cache.store("key1", 1)
        cache.store("key2", 2)
        assert len(cache) == 2

        # Access key1 to make it recently used
        cache.get("key1")

        # Add another item, should evict key2 (least recently used)
        cache.store("key3", 3)
        assert len(cache) == 2

        # key1 and key3 should be present, key2 should be evicted
        assert cache.get("key1") == 1
        assert cache.get("key3") == 3
        assert cache.get("key2") is None

    def test_overwrite_existing_key(self) -> None:
        cache: ResourceCache[str, int] = ResourceCache("test")

        cache.store("key1", 1)
        cache.store("key1", 42)  # Overwrite

        assert len(cache) == 1
        assert cache.get("key1") == 42

    def test_eviction_callback(self) -> None:
        evicted_items = []

        def on_evict(value: int) -> None:
            evicted_items.append(value)

        cache: ResourceCache[str, int] = ResourceCache(
            "test", max_size=2, on_evict=on_evict
        )

        # Fill cache
        cache.store("key1", 1)
        cache.store("key2", 2)

        # This should trigger eviction of key1
        cache.store("key3", 3)

        assert evicted_items == [1]

    def test_clear(self) -> None:
        evicted_items = []

        def on_evict(value: int) -> None:
            evicted_items.append(value)

        cache: ResourceCache[str, int] = ResourceCache("test", on_evict=on_evict)

        # Add some items and generate stats
        cache.store("key1", 1)
        cache.store("key2", 2)
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss

        assert len(cache) == 2
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1

        # Clear the cache
        cache.clear()

        # Should evict all items and reset stats
        assert len(cache) == 0
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0
        assert sorted(evicted_items) == [1, 2]

    def test_clear_without_eviction_callback(self) -> None:
        cache: ResourceCache[str, int] = ResourceCache("test")

        cache.store("key1", 1)
        cache.store("key2", 2)

        cache.clear()

        assert len(cache) == 0
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0

    def test_str_representation(self) -> None:
        cache: ResourceCache[str, int] = ResourceCache("TestCache")

        # Generate some stats
        cache.store("key1", 1)
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss

        result = str(cache)
        expected = "TestCache Cache: 1 hits, 1 misses (50.0% hit rate)"
        assert result == expected

    def test_repr_representation(self) -> None:
        cache: ResourceCache[str, int] = ResourceCache("TestCache", max_size=10)

        cache.store("key1", 1)
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss

        result = repr(cache)
        expected = (
            "<ResourceCache 'TestCache' size=1/10, "
            "stats=1 hits, 1 misses (50.0% hit rate)>"
        )
        assert result == expected

    def test_complex_lru_scenario(self) -> None:
        """Test a more complex LRU scenario with multiple accesses."""
        cache: ResourceCache[str, str] = ResourceCache("test", max_size=3)

        # Fill cache
        cache.store("a", "value_a")
        cache.store("b", "value_b")
        cache.store("c", "value_c")

        # Access items in specific order to establish usage patterns
        cache.get("a")  # a becomes most recent
        cache.get("b")  # b becomes most recent
        # c is now least recent

        # Add new item - should evict c
        cache.store("d", "value_d")

        assert cache.get("a") == "value_a"
        assert cache.get("b") == "value_b"
        assert cache.get("d") == "value_d"
        assert cache.get("c") is None

    def test_generic_types(self) -> None:
        """Test that the cache works with different key and value types."""
        # Integer keys, string values
        int_cache: ResourceCache[int, str] = ResourceCache("int_test")
        int_cache.store(1, "one")
        int_cache.store(2, "two")
        assert int_cache.get(1) == "one"
        assert int_cache.get(2) == "two"

        # Tuple keys, list values
        tuple_cache: ResourceCache[tuple[int, str], list[int]] = ResourceCache(
            "tuple_test"
        )
        tuple_cache.store((1, "a"), [1, 2, 3])
        tuple_cache.store((2, "b"), [4, 5, 6])
        assert tuple_cache.get((1, "a")) == [1, 2, 3]
        assert tuple_cache.get((2, "b")) == [4, 5, 6]

    def test_stats_tracking_accuracy(self) -> None:
        """Test that hit/miss statistics are tracked accurately."""
        cache: ResourceCache[str, int] = ResourceCache("stats_test")

        # Store some items
        cache.store("a", 1)
        cache.store("b", 2)
        cache.store("c", 3)

        # Generate mixed hits and misses
        assert cache.get("a") == 1  # Hit 1
        assert cache.get("missing1") is None  # Miss 1
        assert cache.get("b") == 2  # Hit 2
        assert cache.get("missing2") is None  # Miss 2
        assert cache.get("c") == 3  # Hit 3
        assert cache.get("a") == 1  # Hit 4 (accessing again)

        assert cache.stats.hits == 4
        assert cache.stats.misses == 2
        assert cache.stats.total_lookups == 6
        assert cache.stats.hit_rate == pytest.approx(66.66666666666666)

    def test_eviction_order_with_mixed_operations(self) -> None:
        """Test eviction order with mixed store and get operations."""
        evicted_values = []

        def track_eviction(value: int) -> None:
            evicted_values.append(value)

        cache: ResourceCache[str, int] = ResourceCache(
            "mixed_test", max_size=2, on_evict=track_eviction
        )

        # Fill cache
        cache.store("first", 1)
        cache.store("second", 2)

        # Access first item to make it recent
        cache.get("first")

        # Store third item - should evict "second"
        cache.store("third", 3)
        assert evicted_values == [2]

        # Store fourth item - should evict "first"
        cache.store("fourth", 4)
        assert evicted_values == [2, 1]

        # Only "third" and "fourth" should remain
        assert cache.get("third") == 3
        assert cache.get("fourth") == 4
        assert cache.get("first") is None
        assert cache.get("second") is None
