import numpy as np
import pytest

from scripts.navigate_rgbd import Episode


@pytest.mark.parametrize("second_x,accepted", [(1.08, True), (1.3, False), (None, False)])
def test_confirmation_requires_consistent_depth_from_moved_camera(second_x, accepted):
    class Camera:
        position = np.zeros(2)

        def __init__(self):
            self.path = []
            self.probes = [{}]
            self.confirmations = []
            self.confirmation_steps = 0

        def reachable(self):
            return {(0, 0): None, (1, 0): (0, 0), (0, 1): (0, 0)}

        def center(self, cell):
            return np.array(cell) * .2

        def move_to(self, parents, waypoint, phase):
            assert waypoint == (0, 1)  # perpendicular to the sightline
            self.position = self.center(waypoint)
            self.path.append(phase)

        def turn(self, yaw, phase):
            self.path.append(phase)

        def probe(self, detector, threshold, phase):
            self.probes.append({})
            observed = None if second_x is None else np.array([second_x, .8, 0])
            return None, None, .02, observed

        def add_frame(self, *unused):
            pass

    camera = Camera()
    result = Episode.confirm(camera, None, .012, np.array([1.0, .8, 0]))
    assert (result is not None) == accepted
    assert camera.confirmation_steps == 2
    if accepted:
        np.testing.assert_allclose(result, [1.04, .8, 0])
    assert camera.confirmations[0]["candidate_probe"] == 1
    assert camera.confirmations[0]["probe"] == 2
