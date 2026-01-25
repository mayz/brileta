"""ModernGL backend utilities."""

import moderngl


def get_uniform(program: moderngl.Program, name: str) -> moderngl.Uniform:
    """Get a uniform from a program with proper type narrowing.

    ModernGL's program["name"] returns Uniform | UniformBlock | Attribute | Varying,
    but we typically know we're accessing uniforms. This helper validates and
    narrows the type.

    Args:
        program: The ModernGL program
        name: The uniform name

    Returns:
        The Uniform object

    Raises:
        TypeError: If the member is not a Uniform
    """
    member = program[name]
    if not isinstance(member, moderngl.Uniform):
        raise TypeError(f"{name} is not a Uniform, got {type(member).__name__}")
    return member
