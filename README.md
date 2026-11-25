# Navigation Foundation Model

[![CI](https://github.com/rsasaki0109/navigation-foundation-model/actions/workflows/ci.yml/badge.svg)](https://github.com/rsasaki0109/navigation-foundation-model/actions/workflows/ci.yml)

**Four ways to turn "head over to the chair" into motion, side by side in one simulator.**

<p align="center">
  <img src="docs/media/hero.gif" alt="Four navigation paradigms (VLM + planning, VLA, WM + planning, WAM) solving the same instruction in the same apartment, each with a top-down map, an egocentric camera and a view of the model's internals" width="100%">
  <br><sub>Same episode (held-out seed 35), same instruction, four paradigms. The 3D views are rendered by [Robot Native Engine](https://github.com/rsasaki0109/RobotNativeEngine)'s wgpu PBR renderer from the live simulation state. The inset is the top-down map, and the right column of each panel shows what each model is thinking.</sub>
</p>

<table>
  <tr>
    <td width="50%" align="center">
      <img src="docs/media/vlm.gif" alt="VLM + planning: a grounding model highlights the target in the panorama, the goal is projected into an online occupancy map and an A* path is followed" width="100%"><br>
      <b>VLM + planning</b><br>
      <sub>A language-grounded detector scores every panorama column for the instructed object; the hit is back-projected with depth into an online occupancy map, and a classical A* planner + pure pursuit drives there. No target in view → frontier exploration.</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/media/vla.gif" alt="VLA: a single network maps the panorama and instruction to a distribution over 21 discrete action tokens" width="100%"><br>
      <b>VLA (vision-language-action)</b><br>
      <sub>One network maps pixels + words straight to a distribution over 21 discrete (v, ω) action tokens, RT-style. No map and no planner. It is trained by behaviour cloning and then DAgger.</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="docs/media/wm.gif" alt="WM + planning: CEM samples action sequences, rolls them out in a learned latent world model and executes the lowest-cost plan" width="100%"><br>
      <b>WM + planning (world model + MPC)</b><br>
      <sub>A latent world model (encoder, action-conditioned dynamics, decoder, goal-distance and collision heads) is searched with CEM: 192 imagined action sequences × 4 iterations every step, seeded by a latent policy prior (TD-MPC style). The map shows the sampled plans; the strips are decoded imagined futures.</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/media/wam.gif" alt="WAM: one model predicts an 8-step action chunk together with the future panoramas it expects to see" width="100%"><br>
      <b>WAM (world-action model)</b><br>
      <sub>One trunk jointly predicts an 8-step action chunk <i>and</i> the panoramas it expects to see at t+2…t+8. Predicting the future acts as an auxiliary signal for the actions. It executes 2 actions, then re-plans.</sub>
    </td>
  </tr>
</table>

## Results

Held-out evaluation on 100 episodes (seeds 0–99; training uses seeds ≥ 100 000, so layouts, object placements, start poses and instruction phrasings are unseen):

| Paradigm | Success | SPL | Steps to goal (successes) | Collisions / episode | Params |
| --- | ---: | ---: | ---: | ---: | ---: |
| VLM + planning | **0.99** | 0.90 | 48.8 | **0.00** | 0.11 M (+ classical planner) |
| ↳ grounding by **pretrained CLIP ViT-B/32, zero-shot** (ray-cast views) | 0.80 | 0.59 | 74.1 | 0.10 | 151 M, not trained here |
| ↳ same, CLIP sees RNE-rendered views (40 episodes) | 0.70 | 0.54 | 64.1 | 0.10 | 151 M, not trained here |
| VLA | 0.94 | 0.92 | **38.3** | 0.06 | 0.69 M |
| WM + planning | 0.80 | 0.63 | 60.6 | 0.35 | 0.73 M (+ CEM) |
| ↳ policy prior only (no planning) | 0.87 | — | — | 6.76 | same network |
| WAM | **0.99** | **0.96** | 41.0 | 0.40 | 2.56 M |

`docs/results.json` has per-episode numbers (`python scripts/evaluate.py 100`).

What the numbers say, in this toy setting:

- **VLM + planning** is the most robust and never touches furniture. It is the only method with memory: a map built from depth, plus frontier exploration when the target is hidden behind a wall. The price is speed, because exploring costs steps.
- **A real foundation model drops in zero-shot.** Replacing the trained grounding net with OpenAI's CLIP ViT-B/32 changes nothing else. Every 4 steps the robot renders 8 camera crops around itself, CLIP scores each against the instruction rewritten as a caption (*"take me to the red chair"* becomes *"a photo of a red chair"*) and against background captions, and the same map and A* planner drive. It reaches 0.80 without seeing a single frame of this world in training. It explores more, because CLIP picks the right direction for only about half the crops, and the planner drops a goal once it has stood there 6 steps without arriving. It does better on the flat-coloured ray-cast views than on the RNE renders (0.70): the RNE furniture models are less canonical-looking than "a red chair" (`navfm/clip_vlm.py`, agent kinds `clip_raycast` / `clip`).
- **WAM** matches it on success and takes the most efficient paths (best SPL). It is a reactive policy like the VLA. Predicting the future panoramas along with an 8-step action chunk acts as an auxiliary task that shapes its representation.
- **VLA** is the fastest per episode but fails more often (6 of 100). Its discrete action tokens are coarse, and it has no memory or look-ahead.
- **WM + planning** shows what planning buys and what it costs. The same network's policy prior alone reaches the goal slightly more often (0.87) but bumps into furniture about 20× more (6.8 vs 0.35 collisions per episode). Searching 10 imagined steps ahead with the collision head keeps it clear. The cost is success: when the learned goal-distance head is wrong, CEM optimises toward the error.

## Engineering notes

Getting four learners to work on one simulator mostly meant fixing the data rather than the models:

| Problem | Symptom | Fix |
| --- | --- | --- |
| World model trained on expert demos only | CEM drives into states whose imagined outcome is wrong (model exploitation). Success ~0.2. | Train on the planner's own rollouts (0.39), then on planner-style *counterfactual* action sequences executed in the true simulator, and seed CEM with a latent policy prior (0.79) |
| Expert aimed 0.6 m ahead along the distance field | It cut corners, so DAgger taught the imitators to keep pushing into furniture | Aim at the farthest waypoint in line of sight; turn in place before a collision |
| Expert's turn-in-place direction flipped sign between neighbouring states | WAM's regression head averaged left and right into "drive straight" | Always turn toward increasing clearance |
| Blocked moves froze the robot | Corner deadlocks: 100+ collisions in an episode | Slide along obstacles like Habitat's `allow_sliding` (still counted as a collision) |
| *Tried, did not help:* WM failures that remain (20 of 100: 4 stall just short of the object, 3 trust a hallucinated "almost there", 4 never see the target, 9 run out of time otherwise) | — | A 5-head bootstrap ensemble of distance heads on the frozen model, plus a target-proximity head. Used as the mean, success fell to 0.70; as a disagreement penalty on the original head, it stayed at 0.80 on 40 episodes. Off by default (`WMPlannerAgent.USE_ENS`). Memory or a better-calibrated value probably matter more |
| *Tried, did not help either:* give the world model memory | — | A recurrent, RSSM-style world model (`RecurrentWorldModel`, kind `rwm`): a GRU state folds in each observation, and imagination continues from it. Trained on 24-step sequences, with counterfactual branches that keep their real history. Carrying the state through a whole episode drifted past what it was trained on (success 0.44). Re-filtering a 16-step window each step recovers most of it: success 0.75, SPL 0.49, 7.2 collisions on 100 episodes, still behind the feedforward model above (0.80, SPL 0.63, 0.35 collisions) |
| *…and trained on whole episodes:* the recurrent model's state drifted because the windows it trained on started mid-episode from an empty state | — | Train on whole episodes from their start, with the state carried as at deployment and imagined 8-step branches every 4 steps (`train_rwm_episodes`, kind `rwm_ep`). With expert-mixed DAgger it fell to 0.27: once off the expert's path, its state "believed" it had arrived (predicted 0.1 m, true 4.8 m) and stopped. Pure on-policy rounds, which add its own long failures, raised it to 0.63 (SPL 0.45, 0.3 collisions) after 3 rounds. Three more rounds made it worse (round success 0.60 → 0.55 → 0.38, evaluation 0.34), so the shipped `rwm_ep.pt` is the round-3 model. Memory needs far more, and more diverse, on-policy data than the feedforward model |

### Whole-episode on-policy fine-tuning

`python scripts/train.py rwm-ep-onpolicy` starts with `NAVFM_CKPT/rwm_ep.pt` and writes each checkpoint to a separate `round_NN` directory under `ONPOLICY_OUT` (default `checkpoints/rwm_ep_onpolicy`). The original checkpoint is kept as round 0. The best validation checkpoint is also available as `ONPOLICY_OUT/rwm_ep.pt`, with per-seed results in `validation.json`. Selection uses episode success on seeds 900000–900029, favoring the earlier round on ties. For mixed training, 15 validation seeds are assigned to each layout.

By default it runs five rounds, with expert actions on 10% of rollout steps, a learning-rate multiplier of 0.7 per round, and at most two rounds' worth of replay episodes. Replay is also capped by the number of expert and counterfactual steps. `ONPOLICY_BETA`, `ONPOLICY_LR_DECAY`, and `ONPOLICY_REPLAY` control these settings. Set `NAVFM_CKPT` to the selected run directory and `NAVFM_RESULTS` to a new file when evaluating on seeds 0–99.

## Harder layout, and real physics

### A multi-room house

`NavEnv(seed, "house")` builds three or four rooms. They are separated by full walls with 0.9 m doors, and every start is at least 4 m from the goal along the path. The target is in view at the start in 12.5% of episodes, against 33% in the apartment. Success rate over 100 held-out episodes in each setting:

| Paradigm | House, apartment-trained (zero-shot) | House, fine-tuned on a 50/50 apartment + house mix | Apartment, same mix-trained model |
| --- | ---: | ---: | ---: |
| VLM + planning | 0.91 | 0.91 (no learned motion to retrain) | 0.99 |
| VLA | 0.70 | 0.88 | 0.95 |
| WAM | 0.80 | **0.94** | 0.98 |
| WM + planning | 0.58 | 0.67 (10 collisions/episode) | 0.86 |
| WM with memory (recurrent) | 0.63 | 0.29: on-policy training collapsed again; the apartment model is kept | 0.29 |

- Only the method with an explicit map transfers without retraining. VLM + planning explores frontiers room by room and loses almost nothing. The reactive VLA and WAM, and the world-model planner, drop 10–25 points.
- A little house data fixes the reactive policies. After fine-tuning on the mix, WAM is the best method in the house (0.94), and the apartment numbers do not regress (VLA 0.95, WAM 0.98, WM 0.86).
- The recurrent world model is the only learned method whose zero-shot house score equals its apartment score (0.63 in both), so its memory does carry across rooms. Its on-policy fine-tuning is unstable, though, and fell to 0.29 here just as it did in the apartment.

Numbers are in `docs/results_house_zeroshot.json`, `docs/results_house_trained.json` and `docs/results_house_trained_on_apartment.json`. Retrain with `scripts/train_house.sh`, which writes to `checkpoints/house/`.

### On Robot Native Engine physics

`navfm/rne_physics.py` runs the same episodes in RNE's Rapier physics. It exports the layout as a `.rne.scene.toml` of fixed box obstacles and drives a joint-driven differential-drive robot (wheel radius 6 cm, track 28 cm). Each 0.25 s step is 15 ticks at 60 Hz. Wheel motors, slip, inertia and contacts are therefore RNE's, not the unicycle model the policies were trained on. The expert succeeds in every episode with no collisions. The apartment-trained models, with no retraining, over 100 episodes:

| Paradigm | Python sim | RNE physics | Collisions / episode in RNE |
| --- | ---: | ---: | ---: |
| VLM + planning | 0.99 | 0.97 | 0.0 |
| VLA | 0.94 | 0.98 | 0.7 |
| WAM | 0.99 | 0.90 | 11.2 |
| WM + planning | 0.80 | 0.71 | 7.3 |

The closed-loop methods barely notice the change of dynamics, because they replan every step. WAM executes 2 actions open-loop from each chunk, and the world-model planner relies on dynamics learned from the unicycle model. Wheel slip that neither expected costs both of them success and collisions. Run with `python scripts/evaluate.py 100 --physics rne`. This needs the `rne_py` wheel built from RobotNativeEngine (`maturin build -m crates/rne_py/Cargo.toml`), which provides `DiffDriveSim.from_scene`.

### WAM execution and world-model transfer

The same WAM weights were evaluated with two actions per chunk, replanning every step (`wam_exec1`), and ACT-style temporal ensembling (`wam_act`). Temporal ensembling replans every step and uses an exponentially weighted average of actions predicted for the current step by overlapping chunks.

| WAM execution | Python success | Python collisions / episode |
| --- | ---: | ---: |
| Two actions (default) | 0.99 | 0.39 |
| Replan every step | 0.97 | 1.70 |
| Temporal ensemble | 0.87 | 21.05 |

`docs/results_wam_python.json` contains all 100 episodes per variant. Replanning every step and temporal ensembling both reduced success and increased collisions in the Python simulator.

`python scripts/train.py wm-rne` collects noisy expert episodes and planner rollouts in Rapier from seed 600000 onward. It adapts the latent transition and collision predictors and writes `checkpoints/rne/wm.pt`. Evaluate that checkpoint with `NAVFM_CKPT=checkpoints/rne` and a separate `NAVFM_RESULTS` file.

## The simulator

`navfm/world.py` is a small, deterministic object-goal navigation world:

- **Apartment:** randomly generated, 8 m × 6 m. It has tall partial walls that block sight, low furniture that blocks motion but not sight, and six coloured target objects (chair, sofa, plant, fridge, bed, tv).
- **Robot:** a unicycle-model robot with radius 0.18 m and 0.25 s control steps, commanded by velocity (v, ω). A blocked move slides along the obstacle and counts as a collision.
- **Observation:** a 360° ray-cast panorama of 128 columns × (RGB + two depth layers), plus odometry and a free-form instruction such as *"take me to the red chair"* or *"find the fridge"*.
- **Success:** the robot comes within 0.5 m of the instructed object within 160 steps.
- **Expert:** a privileged expert (geodesic distance field + line-of-sight pure pursuit) produces demonstrations and DAgger labels. It succeeds in 100% of episodes with zero collisions.

The 3D views in the GIFs are for humans. The overview shows the plans as glowing dots: the A* path, the CEM samples and the best plan, or the WAM action chunk. The models only ever see the 128-column panorama strip shown under the egocentric camera.

**Rendering.** `navfm/rne_scene.py` exports every frame as a scene of primitives and [Poly Haven](https://polyhaven.com/models) CC0 furniture models (`assets/polyhaven`, fetched by `scripts/fetch_polyhaven.py`). Each model is fitted to its object's footprint and turned to face the room, and the six instructable objects are drawn in their instruction colour. Robot Native Engine's [`examples/150_scene_replay_render`](https://github.com/rsasaki0109/RobotNativeEngine/tree/main/examples/150_scene_replay_render) renders each frame with wgpu. No GPU is needed: Mesa's llvmpipe software Vulkan driver renders a frame in about 50 ms. If the RNE binary is not found, `make_gifs.py` falls back to a built-in ray caster.

## Quickstart

```bash
pip install -r requirements.txt          # numpy, scipy, pillow, torch (CPU is fine)

python -m pytest -q tests                # simulator, expert, scene export; shipped checkpoints reproduce docs/results.json on the hero seeds
python scripts/evaluate.py 20            # evaluate the shipped checkpoints
# GIFs with RNE rendering (sibling checkout of RobotNativeEngine; on a GPU-less box: apt install mesa-vulkan-drivers)
(cd ../RobotNativeEngine && cargo build --release -p scene_replay_render --example 150_scene_replay_render)
python scripts/make_gifs.py 35           # or RNE_REPLAY_BIN=/path/to/150_scene_replay_render; --raycast for no RNE

# retrain everything from scratch (CPU, ~1-2 h total)
python scripts/collect.py 2500 data/expert.npz
python scripts/train.py vlm wm vla wam
```

## Layout

| Path | What |
| --- | --- |
| `navfm/world.py` | simulator: layouts, ray casting, dynamics, observations |
| `navfm/expert.py` | geodesic-field expert used for demonstrations / DAgger labels |
| `navfm/models.py` | the four networks (shared circular-conv panorama encoder + text encoder) |
| `navfm/agents.py` | deployable agents: grounding + A*, token policy, CEM-MPC, chunked WAM |
| `navfm/train.py`, `scripts/train.py` | training loops, DAgger / on-policy fine-tuning |
| `navfm/render.py`, `navfm/rne_scene.py`, `scripts/make_gifs.py` | hero GIF panels; RNE scene export |
| `checkpoints/` | trained weights used for the GIFs and the table |

## Scope

These models are deliberately tiny (0.1–2.6 M parameters, trained on a CPU) so that the whole pipeline can be rerun and inspected in an afternoon. They implement the architectural *paradigms* behind VLM-based planners, VLAs, world-model planners and world-action models. They are not pretrained foundation models, and the numbers above compare paradigms in this toy world, not published systems.

## License

MIT
