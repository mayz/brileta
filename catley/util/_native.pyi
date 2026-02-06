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
