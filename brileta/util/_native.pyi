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
