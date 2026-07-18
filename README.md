# STREL Policy Optimization for Battery-Aware Multi-Drone Systems

This repository investigates how a neural control policy can be trained directly from **formal spatial-temporal mission specifications**. The case study consists of one or more battery-powered drones moving in a continuous two-dimensional environment. The drones must reach a target region, return to a charging station when energy becomes low, avoid forbidden areas, and—when multiple drones are present—maintain a safe distance from one another.

The central idea is to replace a purely hand-designed reward with the **quantitative robustness** of a STREL/CSTREL specification. Positive robustness means that a generated trajectory satisfies the formal mission; negative robustness means that at least one requirement is violated. The magnitude indicates the satisfaction or violation margin.

The experiments are organized as a progressive sequence. Each case adds one source of complexity, making it possible to study what the policy learns, where performance degrades, and how explicit logical safety constraints change the resulting behavior.

---

## Contents

- [Project overview](#project-overview)
- [Methodology](#methodology)
- [Environment](#environment)
- [Mission specification](#mission-specification)
- [Experimental cases](#experimental-cases)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Running the experiments](#running-the-experiments)
- [Understanding the outputs](#understanding-the-outputs)
- [Limitations and future work](#limitations-and-future-work)

---

## Project overview

The project addresses policy learning for a small **Collective Adaptive System** composed of autonomous drones and static environmental nodes. At every time step, a multilayer perceptron receives the current global state and produces a two-dimensional movement command for each drone.

Rather than optimizing only an instantaneous objective, the policy is trained to generate complete trajectories that satisfy requirements extending across both **space** and **time**. These include conditional mission rules such as:

- when a drone has sufficient battery, it should reach the target zone;
- when its battery becomes low, it should return to the charging base and remain there until recharged;
- it must remain operational throughout the rollout;
- in the safety extensions, it must avoid obstacles and other drones.

The full computational chain is differentiable:

```text
policy parameters
      ↓
policy actions
      ↓
differentiable drone dynamics
      ↓
complete trajectories
      ↓
smooth STREL robustness
      ↓
gradient-based policy update
```

This allows the formal specification itself to become the main learning signal.

---

## Methodology

### Policy

The controller is a feed-forward multilayer perceptron. It receives a flattened representation of the relevant system state and returns one two-dimensional action per drone:

```text
u_t = πθ(x_t)
```

For two drones, the policy produces two movement vectors jointly. Because it observes the global configuration, it can in principle coordinate the agents instead of controlling them independently.

### Differentiable dynamics

Drone motion evolves in continuous space, even though the environment is represented using a grid. The velocity update includes momentum:

```text
v(t+1) = 0.6 v(t) + 0.4 a(t)
```

The position update is:

```text
p(t+1) = clip(p(t) + 0.8 v(t+1), 1, 6)
```

The clipping operation keeps each drone inside the environment boundaries. This means the drones are not restricted to jumping between adjacent grid cells; they can follow smooth continuous trajectories.

### Robustness-based optimization

Let `ρ(Φ, τ)` be the quantitative robustness of trajectory `τ` with respect to specification `Φ`.

- `ρ > 0`: the trajectory satisfies the specification;
- `ρ < 0`: the trajectory violates the specification;
- larger positive values indicate a wider safety or satisfaction margin;
- more negative values indicate a stronger violation.

The main optimization objective is therefore:

```text
maximize ρ(Φ, τθ)
```

or equivalently minimize the negative robustness. The implementation also uses regularization terms to discourage unnecessarily aggressive actions and to improve optimization stability.

Because exact logical minimum and maximum operators are not smooth, differentiable approximations are used during training. This makes it possible to backpropagate the robustness signal through the complete simulated trajectory and into the neural policy.

---

## Environment

### Spatial domain

The environment is a `6 × 6` grid used as a spatial reference structure. Drone coordinates remain continuous within the interval `[1, 6]` along both axes.

### Graph representation

The scene is represented as a fully connected weighted graph. Its nodes include:

1. grid nodes;
2. charging-base nodes;
3. drone nodes.

Every pair of nodes is connected, and each edge is weighted by Euclidean distance. This representation supports STREL spatial operators such as reachability and proximity between different classes of entities.

### Node features

Each node is represented by the six-dimensional vector:

```text
(x, y, vx, vy, b, τ)
```

where:

- `x, y` are position coordinates;
- `vx, vy` are velocity components;
- `b` is the battery state for drone nodes;
- `τ` identifies the node type.

The type labels are:

| Node type | Label | Role |
|---|---:|---|
| Base | `0` | Static charging station |
| Drone | `1` | Mobile controlled agent |
| Grid | `2` | Static spatial reference node |

### Initial conditions

At the beginning of each rollout:

- drone positions are sampled inside the grid;
- drone velocities are initialized to zero;
- each drone starts with full battery, `b = 1.0`;
- the charging base is fixed at approximately `(2.0, 2.0)` in the experimental description;
- the target region is the upper part of the environment, defined by `y > 4.5`.

### Battery and charging

Battery decreases by approximately `2%` at each time step. When a drone is sufficiently close to the charging base, charging can add approximately `10%` per step.

Charging is implemented smoothly as a function of drone-to-base distance. It becomes strong at distances below roughly `0.8` units and gradually vanishes farther away. This smooth formulation preserves differentiability.

### Training setup

The principal experiments use:

| Parameter | Value |
|---|---:|
| Rollout horizon | `70` time steps |
| Training iterations | `1000` |
| Batch size | `64` |
| Mini-batch size | `64` |
| Initial learning rate | `3 × 10⁻⁴` |
| Later learning rate | `5 × 10⁻⁵` after iteration 500 |
| Gradient clipping | `1.0` |
| Optimizer | Adam |

Some exploratory notebooks may expose alternative horizons, evaluation batch sizes, or controller settings.

---

## Mission specification

The core battery-aware mission combines three requirements.

### 1. Reach the target with sufficient battery

When the battery is above the low-energy threshold, the drone should eventually reach the target region within the specified temporal horizon.

```text
high_battery → eventually reach(target)
```

with:

```text
high_battery ≡ b > 0.3
```

### 2. Return to the base when battery is low

When battery is low, the drone should eventually reach the charging base and remain there until sufficiently recharged.

```text
low_battery → eventually(reach(base) until full_battery)
```

with:

```text
low_battery  ≡ b ≤ 0.3
full_battery ≡ b > 0.9
```

### 3. Remain alive

Battery must stay positive throughout the rollout:

```text
alive ≡ b > 0
```

The complete base mission is applied globally:

```text
Φbase = globally(φtarget ∧ φrecharge ∧ φalive)
```

Additional formulas are introduced in the obstacle and collision cases.

---

## Experimental cases

## Case 1 — One drone: base case

Folder: [`1drone/`](1drone/)

This is the simplest experiment and acts as the baseline for the entire project. A single drone must learn a repeating mission cycle:

1. move toward the target area while the battery is sufficiently high;
2. return toward the charging base before energy becomes critical;
3. remain close enough to the base to recharge;
4. leave again and continue operating.

There are no other drones and no forbidden region. Therefore, the experiment isolates the policy's ability to learn the relation between **goal reaching** and **battery management**.

The folder includes training, evaluation, policy-analysis, and test-time notebooks, together with a saved trained policy.

## Case 2 — Two drones: simple multi-agent test

Folder: [`2drones_simple/`](2drones_simple/)

The second case repeats the battery-aware mission with two drones in the same environment. Each drone must satisfy the same target, recharge, and survival requirements.

This case tests whether the shared policy can:

- generate actions for two agents simultaneously;
- coordinate their mission cycles from different random initial positions;
- prevent the performance of one drone from masking failure by the other.

The robustness aggregation is intentionally demanding because collective satisfaction depends on the weaker agent. A trajectory can therefore have a good average behavior while still failing if one drone violates the specification at a particular time.

Importantly, this baseline does **not** yet contain an explicit inter-drone separation rule. Consequently, both drones may learn similar paths and occupy the same area at the same time.

## Case 3 — Two drones with an obstacle

Folder: [`2drones_obstacle/`](2drones_obstacle/)

This experiment introduces a forbidden portion of the grid. Conceptually, it represents an impassable wall or restricted zone located between the lower/base area and the upper target region.

The policy must now solve four tasks at once:

1. reach the target when battery is sufficient;
2. return to the base when battery is low;
3. remain alive;
4. never enter the obstacle.

The additional safety formula is:

```text
φobstacle = not inside(obstacle)
```

and the complete specification becomes:

```text
Φobstacle = globally(
    φtarget ∧ φrecharge ∧ φalive ∧ φobstacle
)
```

A successful trajectory can no longer take the shortest direct path if that path intersects the forbidden region. Instead, the drones must bend their trajectories around it while preserving enough energy to reach the target and return to the base.

This is a materially harder optimization problem because the obstacle reduces the set of feasible trajectories and creates a trade-off between path length, battery consumption, and logical safety.

## Case 4 — Two drones with collision avoidance

Folder: [`no_collision/`](no_collision/)

The simple two-drone experiment revealed an important weakness: without an explicit separation property, the policy has no reason to avoid collisions. The drones may satisfy the mission objective while repeatedly moving too close to one another.

A collision event is counted whenever the Euclidean distance between the two drones falls below `0.2`:

```text
distance(p1, p2) < 0.2
```

The safety requirement added to the specification is:

```text
φseparation = distance(p1, p2) > dsafe
```

and must hold throughout the rollout:

```text
Φcollision = globally(
    φtarget ∧ φrecharge ∧ φalive ∧ φseparation
)
```

This case tests whether formal logic can correct unsafe emergent behavior without replacing the entire learning framework. The two drones are encouraged to preserve distinct paths or timing patterns while still visiting the same target and charging area.

## Case 5 — All constraints together

Folder: [`all_together/`](all_together/)

The final experiment combines every requirement:

- target reaching;
- battery-aware return to base;
- survival;
- obstacle avoidance;
- collision avoidance.

The full specification is:

```text
Φall = globally(
    φtarget
    ∧ φrecharge
    ∧ φalive
    ∧ φobstacle
    ∧ φseparation
)
```

This is the project's stress test. The policy must find trajectories that are simultaneously useful, energy-aware, obstacle-free, and mutually safe. A trajectory that satisfies four requirements but violates only one still has negative full-specification robustness.

---

## Results

### Result summary

| Experiment | Main result | Main weakness |
|---|---|---|
| One drone | Stable target/recharge cycles and consistently positive robustness | Only a single-agent baseline |
| Two drones | Positive average robustness and learned multi-agent mission behavior | More fragile worst-case performance; collisions are not prevented |
| Obstacle | Drones learn to route around the forbidden area while preserving recharge behavior | Lower robustness margin and more temporary violations |
| Collision avoidance | Collision frequency is reduced dramatically while mission behavior is retained | A small number of difficult episodes still violate separation |
| All constraints | The policy can learn meaningful combined behavior and positive average training robustness | Evaluation is close to the satisfaction boundary; worst-case robustness is often negative |

### One-drone result

The single-drone policy learns a clear and repeatable behavioral pattern. Across best, median, and worst evaluation episodes, it performs excursions toward the target region and then returns near the charging base.

The reported mean robustness is concentrated around approximately `0.52`, with mean and median close to one another. Minimum robustness is also generally positive. This is significant because minimum robustness represents the least favorable instant of an episode: positive values indicate that the property is maintained across the trajectory rather than merely satisfied on average.

Battery traces confirm the intended mechanism. Energy decreases away from the base and rises again when the drone returns to recharge. Even weaker episodes retain a comfortable margin above complete depletion.

**Interpretation:** the base mission is learnable and the policy generalizes consistently across different initial drone positions.

### Two-drone result

The two-drone policy preserves the same broad mission pattern. Both agents generally travel toward the target and return to recharge. Mean robustness remains concentrated in the positive region, showing that average collective behavior is successful in most episodes.

However, minimum robustness reveals a more fragile result. A non-negligible subset of episodes becomes negative at one or more time instants. The reason is structural: the full collective property is constrained by the weaker drone, so one poorly timed return or incomplete target-reaching behavior can invalidate the episode.

**Interpretation:** the method scales from one to two agents, but adding agents reduces the safety margin and exposes weaknesses that average robustness alone can hide.

### Obstacle-avoidance result

In the best and median episodes, the drones visibly alter their paths to go around the forbidden region rather than crossing it. The recharge cycle remains active, demonstrating that the policy does not solve obstacle avoidance by abandoning the energy-management part of the task.

Mean robustness becomes positive after a difficult initial phase and remains positive for most of the rollout. Nevertheless, its plateau is lower and its spread is wider than in the obstacle-free experiment. The minimum-robustness distribution contains a substantial negative portion, with an overall mean reported as slightly negative.

Training also becomes less smooth, with stronger oscillations in robustness and loss.

**Interpretation:** obstacle-aware behavior is learned, but the obstacle substantially narrows the feasible operating margin. Average success should not be mistaken for strict satisfaction in every episode.

### Collision-avoidance result

The original two-drone scenario has a serious safety problem. Using a collision threshold of `0.2`, the reported evaluation observed:

- approximately **17.89 collisions per episode on average**;
- up to **33 collisions in one episode**;
- at least one collision in every analyzed episode.

After adding the explicit separation property, the reported figures become:

- approximately **0.02 collisions per episode on average**;
- a maximum of **3 collisions in one episode**;
- many trajectories with no collision during the full rollout.

The reduction from `17.89` to `0.02` collisions per episode is approximately **99.9%**.

The trajectories show that the drones generally maintain distinct paths while preserving target-reaching and recharge cycles. Mean robustness remains positive and training is comparatively stable. Some negative minimum-robustness episodes remain, so the result is not a formal guarantee of zero collisions under every initial condition.

**Interpretation:** explicitly encoding inter-agent safety in the logical objective is highly effective. The experiment also demonstrates why mission completion alone is an insufficient training objective for safety-critical multi-agent systems.

### Full-specification result

The combined experiment is the hardest setting. The best episodes demonstrate that the complete mission is feasible: both drones can reach the target, avoid the obstacle, remain separated, and continue using the charging base.

The median and worst episodes are substantially weaker. Reported robustness over time begins negative, gradually improves, and becomes positive only later in the rollout. The mean robustness distribution lies close to zero: its mean is only slightly positive, while its median is slightly negative. Minimum robustness is clearly negative for many episodes.

Training still improves the policy and reaches a positive average-robustness regime, but evaluation shows that this success is not uniformly reliable across initial conditions.

**Interpretation:** the framework can optimize a rich conjunction of mission and safety requirements, but the present policy architecture and training setup are close to their practical limit in the fully constrained scenario. Positive average robustness is not enough to claim robust satisfaction of the complete specification.

---

## Repository structure

```text
STREL_policy_optimization-main/
├── README.md
├── 1drone/
│   ├── training.ipynb
│   ├── evaluation.ipynb
│   ├── evaluation_mppi.ipynb
│   ├── policy_analysis.ipynb
│   ├── test_time_notebook.ipynb
│   ├── run_policy_analysis.py
│   ├── _compare_policies.py
│   ├── training_methodology.md
│   └── policy_1.pt
├── 2drones_simple/
│   ├── 2drones.ipynb
│   ├── policy_2.pt
│   ├── policy_best_2.pt
│   ├── policy_checkpoint_2.pt
│   └── best_trajectory.gif
├── 2drones_obstacle/
│   ├── 2drones.ipynb
│   ├── 2drones_correct.ipynb
│   ├── policy_2.pt
│   ├── policy_best_2.pt
│   ├── policy_checkpoint_2.pt
│   └── training-preview and trajectory GIFs
├── no_collision/
│   ├── 2drones_nocollision.ipynb
│   ├── policy_2.pt
│   ├── policy_best_2.pt
│   ├── policy_checkpoint_2.pt
│   └── training-preview and trajectory GIFs
├── all_together/
│   ├── 2drone_all.ipynb
│   ├── policy_2.pt
│   ├── policy_best_2.pt
│   ├── policy_checkpoint_2.pt
│   └── training-preview and trajectory GIFs
├── Presentazione_8min 2.pptx
└── Presentazione_8min 3.pdf
```

The one-drone notebooks rely on reusable local modules such as `drone`, `policy`, `stl`, and `training`. Ensure those modules are present on the Python path in the repository version being executed. The two-drone notebooks contain more of the experimental implementation directly inside the notebooks.

---

## Installation

A pinned `requirements.txt` is not currently included, so the following installation command reflects the packages imported by the repository notebooks:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell

python -m pip install --upgrade pip
pip install \
  torch \
  torch-geometric \
  numpy \
  pandas \
  matplotlib \
  seaborn \
  tqdm \
  scikit-learn \
  pillow \
  jupyter
```

PyTorch and PyTorch Geometric installation can depend on the operating system, CUDA version, and hardware. For GPU execution, install the builds appropriate for the local CUDA environment.

Then launch Jupyter from the repository root:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

---

## Running the experiments

The repository is notebook-oriented. Run each notebook from top to bottom so that configuration classes, logic operators, policy definitions, training functions, and evaluation utilities are defined in the intended order.

### One drone

Open:

```text
1drone/training.ipynb
```

Use it to train the single-drone policy. Then evaluate the saved policy with:

```text
1drone/evaluation.ipynb
```

Additional analysis is available in:

```text
1drone/policy_analysis.ipynb
1drone/test_time_notebook.ipynb
1drone/evaluation_mppi.ipynb
```

### Two-drone baseline

Open and run:

```text
2drones_simple/2drones.ipynb
```

This notebook contains the simple two-agent battery-aware experiment.

### Obstacle avoidance

Open and run:

```text
2drones_obstacle/2drones_correct.ipynb
```

The folder also contains an earlier `2drones.ipynb`; the `2drones_correct.ipynb` filename indicates the corrected experimental version and should generally be preferred.

### Collision avoidance

Open and run:

```text
no_collision/2drones_nocollision.ipynb
```

This notebook includes the inter-drone separation property and collision counting.

### All constraints

Open and run:

```text
all_together/2drone_all.ipynb
```

This trains and evaluates the complete conjunction of mission, battery, obstacle, and separation requirements.

### Saved policies

The `.pt` files are PyTorch model checkpoints:

- `policy_checkpoint_2.pt`: intermediate or resumable checkpoint;
- `policy_best_2.pt`: checkpoint associated with the best recorded training performance;
- `policy_2.pt`: final or selected policy artifact, depending on the notebook.

Exact loading behavior is defined inside each notebook. Architecture and configuration values must match those used when the checkpoint was saved.

---

## Understanding the outputs

The notebooks generate several complementary evaluation views.

### Trajectory plots

These show the motion of each drone in the grid, usually for the best, median, and worst episodes. They should be read together with the target region, base position, and obstacle geometry.

Questions to ask:

- Does the drone actually enter the target region?
- Does it return to the base before battery becomes low?
- Does the path cross the obstacle?
- Do two drones collapse onto the same trajectory?

### Robustness over time

This plot shows how specification satisfaction evolves during the rollout.

- a positive curve indicates satisfaction at that time;
- a negative curve indicates a violation;
- wide percentile or standard-deviation bands indicate sensitivity to initial conditions;
- an initially negative curve followed by recovery means the policy reaches favorable states later but does not satisfy the global property uniformly from the beginning.

### Robustness during training

This plot measures whether policy optimization is improving the logical objective.

A transition from negative to positive robustness indicates that the learned trajectories move from violation to satisfaction. Oscillations or repeated drops suggest a more difficult optimization landscape.

### Training loss

The loss combines negative robustness and regularization terms. A negative final loss is consistent with positive robustness dominating the auxiliary penalties, but it should not be interpreted without checking evaluation robustness.

### Mean-robustness distribution

Mean robustness summarizes the average satisfaction margin across an episode. It is useful for comparing overall performance across random initial states.

### Minimum-robustness distribution

Minimum robustness is the stricter metric. It reports the least favorable instant of each episode. An episode may have positive mean robustness but negative minimum robustness, meaning that it performs well overall yet violates at least one requirement temporarily.

For safety-critical interpretation, minimum robustness and explicit violation counts are more informative than mean robustness alone.

### Battery traces

Battery plots reveal whether the intended recharge mechanism has actually been learned. A meaningful policy should show:

- discharge while moving away from the base;
- recharge near the base;
- repeated mission/recharge cycles;
- no approach to complete depletion.

### Collision counts

Collision counts are computed from pairwise Euclidean distances. They provide a concrete safety metric in addition to logical robustness. This is particularly useful because a policy can exhibit positive average mission robustness while still producing physically unsafe interactions.

---

## Limitations and future work

The experiments demonstrate the usefulness of robustness-driven policy optimization, but they also expose several limitations.

1. **No universal safety guarantee.** Positive empirical robustness over sampled episodes does not prove satisfaction for every possible initial condition.
2. **Small system scale.** The experiments use one or two drones and a fixed-size environment. Larger fleets may require graph neural networks, attention-based decentralized policies, or parameter sharing.
3. **Centralized state input.** The MLP uses a global view of the environment. Real drones may have partial observations, communication delays, or sensor noise.
4. **Simplified dynamics.** The model does not capture full aerial vehicle dynamics, wind, actuator limits, uncertainty, or three-dimensional motion.
5. **Soft logical semantics.** Smooth approximations improve optimization but may differ from exact hard robustness near specification boundaries.
6. **Combined-task fragility.** The final experiment shows that adding several hard requirements can leave the learned policy close to zero robustness.
7. **Reproducibility infrastructure.** The repository would benefit from fixed random seeds, a pinned dependency file, command-line training scripts, exported numerical result tables, and automated tests.

Promising extensions include curriculum learning across the five scenarios, constraint-specific weighting, adversarial initial-state sampling, formal post-training verification, model-predictive correction at test time, and architectures designed explicitly for a variable number of agents.
