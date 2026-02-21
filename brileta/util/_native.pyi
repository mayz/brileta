class _NoiseState:
    seed: int
    frequency: float
    def __init__(
        self,
        seed: int = 1337,
        noise_type: int = 0,
        frequency: float = 0.01,
        fractal_type: int = 0,
        octaves: int = 3,
        lacunarity: float = 2.0,
        gain: float = 0.5,
        weighted_strength: float = 0.0,
        ping_pong_strength: float = 2.0,
        cellular_distance_func: int = 1,
        cellular_return_type: int = 1,
        cellular_jitter_mod: float = 1.0,
        domain_warp_type: int = 0,
        domain_warp_amp: float = 1.0,
    ) -> None: ...
    def sample_2d(self, x: float, y: float) -> float: ...
    def sample_3d(self, x: float, y: float, z: float) -> float: ...
    def domain_warp_2d(self, x: float, y: float) -> tuple[float, float]: ...
    def domain_warp_3d(
        self, x: float, y: float, z: float
    ) -> tuple[float, float, float]: ...

class WFCContradictionError(Exception): ...

def astar(
    cost: object,
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
) -> list[tuple[int, int]]: ...
def fov(
    transparent: object,
    visible: object,
    origin_x: int,
    origin_y: int,
    radius: int,
) -> None: ...
def wfc_solve(
    width: int,
    height: int,
    num_patterns: int,
    propagation_masks: object,
    pattern_weights: object,
    initial_wave: object,
    seed: int,
) -> list[list[int]]: ...
