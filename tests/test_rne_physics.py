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
