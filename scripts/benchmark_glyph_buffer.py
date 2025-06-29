import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

from tcod.console import Console

from catley import colors
from catley.util.glyph_buffer import GlyphBuffer

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))


class Benchmark:
    def __init__(self, name: str):
        self.name = name

    def setup(self, width: int, height: int) -> Any:
        raise NotImplementedError

    def run(self, instance: Any, iterations: int):
        raise NotImplementedError


class TCODConsoleBenchmark(Benchmark):
    def __init__(self):
        super().__init__("tcod.console.Console")

    def setup(self, width: int, height: int) -> Console:
        return Console(width, height)

    def run_clear(self, instance: Console, iterations: int):
        for _ in range(iterations):
            instance.clear(ch=ord(" "), fg=colors.WHITE, bg=colors.BLACK)

    def run_put_char(self, instance: Console, iterations: int):
        for _ in range(iterations):
            x = random.randint(0, instance.width - 1)
            y = random.randint(0, instance.height - 1)
            instance.ch[y, x] = ord("@")
            instance.fg[y, x] = colors.WHITE
            instance.bg[y, x] = colors.BLACK

    def run_print(self, instance: Console, iterations: int):
        for _ in range(iterations):
            x = random.randint(0, instance.width - 10)
            y = random.randint(0, instance.height - 1)
            instance.print(x, y, "Hello, world!", fg=colors.WHITE, bg=colors.BLACK)

    def run_draw_frame(self, instance: Console, iterations: int):
        for _ in range(iterations):
            instance.draw_frame(
                0, 0, instance.width, instance.height, fg=colors.WHITE, bg=colors.BLACK
            )


class GlyphBufferBenchmark(Benchmark):
    def __init__(self):
        super().__init__("GlyphBuffer")

    def setup(self, width: int, height: int) -> GlyphBuffer:
        return GlyphBuffer(width, height)

    def run_clear(self, instance: GlyphBuffer, iterations: int):
        for _ in range(iterations):
            instance.clear(ch=ord(" "), fg=(255, 255, 255, 255), bg=(0, 0, 0, 255))

    def run_put_char(self, instance: GlyphBuffer, iterations: int):
        for _ in range(iterations):
            x = random.randint(0, instance.width - 1)
            y = random.randint(0, instance.height - 1)
            instance.put_char(x, y, ord("@"), (255, 255, 255, 255), (0, 0, 0, 255))

    def run_print(self, instance: GlyphBuffer, iterations: int):
        for _ in range(iterations):
            x = random.randint(0, instance.width - 10)
            y = random.randint(0, instance.height - 1)
            instance.print(
                x, y, "Hello, world!", fg=(255, 255, 255, 255), bg=(0, 0, 0, 255)
            )

    def run_draw_frame(self, instance: GlyphBuffer, iterations: int):
        for _ in range(iterations):
            instance.draw_frame(
                0,
                0,
                instance.width,
                instance.height,
                fg=(255, 255, 255, 255),
                bg=(0, 0, 0, 255),
            )


def run_benchmarks(
    benchmarks: list[Benchmark],
    test_name: str,
    iterations: int,
    width: int,
    height: int,
):
    print(f"--- {test_name} ---")
    for benchmark in benchmarks:
        instance = benchmark.setup(width, height)
        test_method = getattr(benchmark, f"run_{test_name.lower().replace(' ', '_')}")

        # Warm-up
        test_method(instance, iterations // 10)

        start_time = time.perf_counter()
        test_method(instance, iterations)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_time_ms = (total_time / iterations) * 1000
        ops_per_sec = iterations / total_time

        print(
            f"  {benchmark.name:<20} | {avg_time_ms:.6f} ms/op | "
            f"{ops_per_sec:,.2f} ops/sec"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GlyphBuffer vs. tcod.console.Console."
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=10000,
        help="Number of iterations for each test.",
    )
    parser.add_argument(
        "--width", type=int, default=80, help="Width of the console/buffer."
    )
    parser.add_argument(
        "--height", type=int, default=50, help="Height of the console/buffer."
    )
    args = parser.parse_args()

    benchmarks = [TCODConsoleBenchmark(), GlyphBufferBenchmark()]

    print(
        f"Running benchmarks with {args.iterations} iterations on a "
        f"{args.width}x{args.height} surface.\n"
    )

    run_benchmarks(benchmarks, "Clear", args.iterations, args.width, args.height)
    run_benchmarks(benchmarks, "Put Char", args.iterations, args.width, args.height)
    run_benchmarks(benchmarks, "Print", args.iterations, args.width, args.height)
    run_benchmarks(benchmarks, "Draw Frame", args.iterations, args.width, args.height)


if __name__ == "__main__":
    main()
