"""navfm: four navigation foundation-model paradigms in one tiny simulator.

* VLM + planning  -- language-grounded perception picks a goal, a classical planner drives.
* VLA             -- vision-language-action policy maps pixels + words straight to actions.
* WM + planning   -- a learned latent world model is searched with CEM (model-predictive control).
* WAM             -- a world-action model jointly predicts future observations and an action chunk.
"""

__version__ = "0.1.0"
