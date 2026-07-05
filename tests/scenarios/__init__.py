"""End-to-end scenario tests driven through the headless SimHarness.

These boot a real, seeded game world and drive it the way a player would
(walk, attack, talk), then assert on resulting game state and the message log.
They exercise systems wired together, complementing the unit tests under
``tests/game/**`` that cover each module in isolation.
"""
