from catley.game.actions.combat import AttackIntent
from catley.game.actions.executors.combat import AttackExecutor


def test_attack_classes_exist() -> None:
    """Verify the new classes can be imported and instantiated."""
    _ = AttackIntent
    _ = AttackExecutor()
    print("✅ Classes imported successfully")


if __name__ == "__main__":
    test_attack_classes_exist()
