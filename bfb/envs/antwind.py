"""The paper's Ant XML and wind parameterization, without temporary XML rewriting.

XML source: maxsbob/BeliefConditionedFB, tag v1.0-paper-original (Apache-2.0).
Wind i = 4 * (cos(2*pi*i/20), sin(2*pi*i/20), 0), viscosity = 0.05.
"""

from pathlib import Path

import numpy as np


def make_antwind(context_id, **kwargs):
    from gymnasium.envs.mujoco.ant_v5 import AntEnv

    if not 0 <= context_id < 20:
        raise ValueError("AntWind context ID must be in [0, 20)")
    env = AntEnv(xml_file=str(Path(__file__).parent / "assets" / "ant.xml"), **kwargs)
    angle = 2 * np.pi * context_id / 20
    env.model.opt.wind[:] = [4 * np.cos(angle), 4 * np.sin(angle), 0]
    env.model.opt.viscosity = 0.05
    return env
