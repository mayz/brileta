from catley.view.render.effects.effects import (
    ExplosionEffect,
    SmokeCloudEffect,
    create_combat_effect_context,
    create_environmental_effect_context,
)
from catley.view.render.effects.environmental import EnvironmentalEffectSystem
from catley.view.render.effects.particles import SubTileParticleSystem


def test_smoke_cloud_creates_environmental_effect() -> None:
    ps = SubTileParticleSystem(10, 10)
    es = EnvironmentalEffectSystem()
    ctx = create_environmental_effect_context(ps, es, 5, 5)
    SmokeCloudEffect().execute(ctx)
    assert len(es.effects) == 1
    assert es.effects[0].effect_type == "smoke"
    assert ps.active_count == 0


def test_explosion_creates_flash_and_debris() -> None:
    ps = SubTileParticleSystem(10, 10)
    es = EnvironmentalEffectSystem()
    ctx = create_environmental_effect_context(ps, es, 0, 0, intensity=1.0)
    ExplosionEffect().execute(ctx)
    assert len(es.effects) == 1
    assert es.effects[0].effect_type == "explosion_flash"
    assert ps.active_count > 0


def test_effect_context_includes_systems() -> None:
    ps = SubTileParticleSystem(5, 5)
    es = EnvironmentalEffectSystem()
    ctx = create_combat_effect_context(ps, es, 0, 0, 1, 1, damage=5)
    assert ctx.particle_system is ps
    assert ctx.environmental_system is es
