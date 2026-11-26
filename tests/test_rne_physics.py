import pytest

from navfm.expert import expert_action
from navfm.rne_physics import RneNavEnv


@pytest.mark.parametrize("seed", [0, 35, 99])
def test_rne_expert_reaches_goal_without_collisions(seed):
    rne_py = pytest.importorskip("rne_py")
    if not hasattr(getattr(rne_py, "DiffDriveSim", None), "from_scene"):
        pytest.skip("Requires an rne_py wheel with DiffDriveSim.from_scene")
    env = RneNavEnv(seed)
    while not env.done:
        env.step(expert_action(env))
    assert env.success
    assert env.collisions == 0


def test_rne_reports_floor_and_obstacle_contact_pairs(tmp_path):
    rne_py = pytest.importorskip("rne_py")
    if not hasattr(getattr(rne_py, "DiffDriveSim", None), "contact_pairs"):
        pytest.skip("Requires the RNE named-contact Python API")

    import numpy as np
    from scripts.simulate_rgbd import create_sim

    obstacle = np.zeros((10, 3), dtype=bool)
    obstacle[8, :] = True
    sim = create_sim(tmp_path, obstacle, np.array([0.0, -0.075]), (0.0, 0.0))
    sim.reset()
    hits = set()
    for _ in range(350):
        sim.step(5.0, 5.0)
        assert len(sim.contact_pairs) == sim.contact_count
        hits.update(tuple(sorted((a, b))) for a, b, impulse in sim.contact_pairs if impulse > 0)
    assert ("ground", "left_wheel") in hits
    assert any("scan_0" in pair and "ground" not in pair for pair in hits)
