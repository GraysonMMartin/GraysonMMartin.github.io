---
layout: page
title: RL for High-Performance Jumping
description: Used a curriculum learning framework to teach simulation quadrupeds to jump. Worked on and written with Aryan Naveen and Pranay Varada.
img: assets/img/state_space.png
importance: 1
category: work
related_publications: true
---

## Abstract
Agents acting in dynamic environments must be able to adapt to the conditions of these environments. Quadrupedal robots are particularly useful for simulating agents in these environments because of their flexibility and real‑world applicability. This reinforcement learning project aims to teach a quadruped to jump over a moving obstacle, using Proximal Policy Optimization (PPO) and a two‑stage curriculum learning framework. Stage I focuses on achieving the ideal jumping motion, while Stage II integrates dynamic obstacles, utilizing a Markov Decision Process (MDP) modified to include motion dynamics and obstacle detection. Reference state initialization (RSI) and domain randomization are employed to enhance robustness and generalization. Simulations in IsaacGym demonstrated that two‑stage curriculum learning improved upon direct training in minimizing obstacle collisions, particularly for obstacles at closer ranges. Timing remains a challenge when attempting to avoid obstacles initialized from farther away. This work highlights the value of curriculum learning methods in RL for robotic tasks and proposes future improvements in reward shaping and learning strategy for enhanced adaptability in dynamic environments.

---

## 1 Introduction
Throughout the field of reinforcement learning, there is significant demand for creating strategies to learn an optimal policy that can achieve a desired reward across changing environments. Furthermore, it is essential that such a policy can *understand* the changes in these environments and act accordingly. After all, many environments in the real world are non‑stationary, with varying conditions in weather and terrain acting on agents, for example. Adaptive policies are also suitable for long‑term success because of their higher levels of robustness. In use cases from search‑and‑rescue operations to industrial inspections, such adaptability can be critical.

Our project aims to solve this problem of understanding changing environments through teaching a quadruped to react to an obstacle moving towards it with a random angle and velocity. Quadrupedal robots such as Boston Dynamics' Spot and ANYbotics' ANYmal are particularly advantageous for undertaking such a project for three reasons. First, there is extensive literature on reinforcement learning for quadruped locomotion, which means that we can iterate on different implementations of RL strategies in order to solve new problems. Specifically, while quadrupedal jumping may be well studied, our project aims to understand how an agent can react to random timing and position. Second, simulations of quadruped movement are both high‑dimensional and visually easy to comprehend, making it clear what the agent has learned once the training process is complete. Third, there is a strong basis for quadruped simulation and translation to real‑world situations where changing environments may be at play; a desire to understand environments motivated our decision to work on this project.

To tackle the particular scenario of jumping over a moving obstacle, we consider several RL techniques. We begin with Proximal Policy Optimization (PPO) as the foundational approach, and subsequently integrate curriculum learning strategies into the training process to systematically optimize the quadruped's ability to master this (perhaps surprisingly) complex task. Curriculum learning breaks this task down into sub‑problems, such that the quadruped first learns how to jump, and once it has mastered that skill, learns how to jump over a moving obstacle.

The remainder of the paper is structured as follows: Sections 2 and 3 detail the existing theory and literature that are pertinent to the experiments we carried out in this report; Section 4 details the modified MDP for our desired problem setting and task of jumping over dynamic obstacles. Sections 5 and 6 detail the selected approaches to solving this problem based on the pre‑existing literature. Finally, in Section 7 we detail the simulated results we observed and evaluate the learned behavior’s performance for various approaches.

---

## 2 Preliminaries
Proximal Policy Optimization (PPO) is a policy‑gradient method that improves upon previous methods in the literature through its relative ease of implementation—requiring just first‑order optimization—and greater robustness relative to optimization techniques such as Trust Region Policy Optimization (TRPO) {% cite schulman2017proximalpolicyoptimizationalgorithms %}. While the objective function maximized by TRPO is the expectation of the product of the advantage and a probability ratio measuring the change between the new and old policy at an update, PPO’s objective function **clips** the probability ratio in this surrogate objective in order to prevent the policy from making unstable updates while simultaneously continuing to allow for exploration. Clipping also avoids having to constrain the KL divergence, making the process computationally simpler and enabling policy updates over multiple epochs. PPO’s simplicity and stability make it a commonplace strategy for finding the optimal policy in RL, which is why we use it as a baseline from which we search for possible improvements, namely curriculum learning methods.

---

## 3 Literature Review
Atanassov, Ding, Kober, Havoutis, and Santina use curriculum learning to stratify the problem of quadrupedal jumping into different sub‑tasks, in order to demonstrate that reference trajectories of mastered jumping are not necessary for learning the optimal policy in this scenario {% cite atanassov2024curriculumbasedreinforcementlearningquadrupedal %}. This increases the adaptability of such a policy, because it is learned by the robot on its own, enabling it to generalize better to unseen real‑world scenarios. Another important component of achieving the optimal policy is reward shaping, which Kim, Kwon, Kim, Lee, and Oh tackle in a stage‑wise fashion in the context of a humanoid backflip {% cite kim2024stagewiserewardshapingacrobatic %}. By developing customized reward and cost definitions for each element of a successful backflip, a complex maneuver like this is segmented into an intuitive fashion that translates well to real‑world dynamics.

---

## 4 Problem Formulation
We utilize a Markov Decision Process (MDP) as the underlying sequential decision‑making model for our RL problem. The MDP is described as a tuple  
$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, r, P, \rho, \gamma)
$$  
where:
- $\mathcal{S}$ is the state space  
- $\mathcal{A}$ is the action space  
- $r: \mathcal{S}\times\mathcal{A}\to\mathbb{R}$ is the reward function  
- $P: \mathcal{S}\times\mathcal{A}\to\Delta(\mathcal{S})$ is the transition operator  
- $\rho\in\Delta(\mathcal{S})$ is the initial distribution  
- $\gamma\in(0,1)$ is the discount factor  

The overall objective of reinforcement learning is to find a policy $\pi:\mathcal{S}\to\mathcal{A}$ that maximizes the cumulative infinite‑horizon discounted reward:  
$$
\mathbb{E}_{s_0\sim\rho,\pi}\Bigl[\sum_{i=0}^\infty \gamma^i\,r(s_i,a_i)\,\Bigm|\,s_0\Bigr].
$$  
A given policy $\pi$ has a value function under transition dynamics $P$ defined as  
$$
V^\pi_P(s)=\mathbb{E}_\pi\Bigl[\sum_{i=0}^\infty \gamma^i\,r(s_i,a_i)\mid s_0=s\Bigr],
$$  
and the state‑action value function is similarly defined as  
$$
Q^\pi_P(s,a)=\mathbb{E}_\pi\Bigl[\sum_{i=0}^\infty \gamma^i\,r(s_i,a_i)\mid s_0=s,\;a_0=a\Bigr].
$$

### 4.1 Quadruped Jumping Obstacle Avoidance MDP
As outlined in Section 1, in this report we extend the work of {% cite atanassov2024curriculumbasedreinforcementlearningquadrupedal %} to enable a quadruped to jump over dynamic obstacles. This enhancement requires not only integrating curriculum learning, but also modifying the underlying MDP formulation to account for the quadruped’s perception module’s feedback as shown in Figure 1.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/quad_controller.png" title="Hierarchical control framework for a quadruped robot" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Overview of the hierarchical control framework for a quadruped robot. The trained policy $\pi$ yields desired joint position deviations from the nominal joint positions that are then fed into a low‑level PD controller producing necessary torques $\tau$ for each joint.
</div>

#### State Space
Building on {% cite atanassov2024curriculumbasedreinforcementlearningquadrupedal %}, we leverage a memory of previous observations and actions to enable the agent to implicitly reason about its own dynamics. We concatenate over a window of $N$ timesteps:
- Base linear velocity $\mathbf{v}\in\mathbb{R}^{3\times N}$  
- Base angular velocity $\boldsymbol{\omega}\in\mathbb{R}^{3\times N}$  
- Joint positions $\mathbf{q}\in\mathbb{R}^{12\times N}$ and velocities $\dot{\mathbf{q}}\in\mathbb{R}^{12\times N}$  
- Previous actions $\mathbf{a}_{t-1}\in\mathbb{R}^{12\times N}$  
- Base orientation $\bar{q}\in\mathbb{R}^{4\times N}$  
- Foot contact states $\mathbf{c}\in\mathbb{R}^{4\times N}$  

To handle dynamic obstacles, we add obstacle detection flags $i^\zeta$ and obstacle endpoints $\zeta$—each also tracked over $N$ frames—resulting in a total dimension of $60N$.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/state_space.png" title="State representation for jumping over dynamic obstacles" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Overview of the state representation used for training our jumping over dynamic obstacle policies. The state vector $s_t\in\mathbb{R}^{60N}$ is constructed by concatenating perception states (obstacle detection and endpoints) and robot states over the past $N$ timesteps.
</div>

#### Action Space
As is standard, the policy outputs deviations $\Delta\mathbf{q}\in\mathbb{R}^{12}$ from nominal joint positions $\mathbf{q}^{\mathrm{nom}}\in\mathbb{R}^{12}$, which are filtered, scaled, and then passed to a low‑level PD controller.

#### Reward
Inspired by {% cite atanassov2024curriculumbasedreinforcementlearningquadrupedal %}, we define
$$
r_{\mathrm{total}} = r^+ \exp\!\Bigl(-\|\;r^-\;\|^2/\sigma\Bigr)^4,
$$
where positive components $r^+$ are multiplied by an exponential penalty on the negative components $r^-$.  
- **Dense rewards:** tracking flight velocity, squat height, foot clearance, and penalizing energy use.  
- **Sparse rewards:** episode‑level bonuses for successful jumps and penalties for fall‑over, collision, or large orientation errors.

---

## 5 State Initialization & Domain Randomization
Static start states can hinder exploration. Peng *et al.*’s Reference State Initialization (RSI) {% cite Peng_2018 %} samples initial states from an expert trajectory. In our work we use a modified RSI: for Stage I we uniformly sample height and $z$‑velocity as in Table 1.

| State Variable       | Min  | Max  |
|----------------------|------|------|
| Height (m)           | 0    | 0.3  |
| $z$‐velocity (m/s)   | -0.5 | 3    |

*Table 1: Initialization ranges for Stage I*  

Stage II adds obstacle randomization (Table 2):

| State Variable                        | Min                                     | Max                                     |
|---------------------------------------|-----------------------------------------|-----------------------------------------|
| Obstacle distance $r$ (m)             | 3                                       | 7                                       |
| Obstacle direction $\theta$ (rad)     | 0                                       | $2\pi$                                  |
| Obstacle orientation $\gamma$ (rad)   | $\frac{\pi}{2}-\theta-\frac{\pi}{3}$    | $\frac{\pi}{2}-\theta+\frac{\pi}{3}$    |
| Obstacle velocity (m/s)               | 3.5                                     | 7                                       |

*Table 2: Initialization ranges for Stage II*  

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/obstacle_init.png" title="Obstacle initialization properties" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
State‑space variables related to the obstacle.
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/rsi_plot.png" title="Stage I training with and without RSI" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Stage I training performance, demonstrating the impact of RSI.
</div>

Domain randomization—varying friction, masses, latencies, etc.—further improves sim‑to‑real generalization {% cite tobin2017domainrandomizationtransferringdeep %} and was adopted from {% cite atanassov2024curriculumbasedreinforcementlearningquadrupedal %}.

---

## 6 Obstacle Avoidance Curriculum Learning
Curriculum learning [5, 6] presents tasks in increasing order of difficulty. In quadruped jumping {% cite atanassov2024curriculumbasedreinforcementlearningquadrupedal %}, Stage I teaches jumps in place, Stage II teaches positional jumps, and Stage III jumps onto platforms. We adapt this to:

1. **Stage I (5 k iters):** learn to jump in place (no obstacle).  
2. **Stage II (15 k iters):** introduce flying obstacle, add collision penalty  
   $$
   r_{\mathrm{col}} = \mathbbm{1}\bigl\{\min_{f\in\text{feet}}\|\mathbf{f}-\zeta\|\le0.1\bigr\}.
   $$
3. **No revisit to Stage I**—we reduce the squat reward scale from 5 to 1 to prioritize obstacle avoidance over ideal form.

---

## 7 Experiments

### 7.1 Training Environment
We use NVIDIA IsaacGym for massive parallelism—thousands of simulenvs on one GPU—and seamless PyTorch integration.

### 7.2 Network Architecture
Actor and critic both: FC layers of sizes $|s|$–512–512–(12 or 1), ReLU activations, $\tanh$‑normalized outputs.

### 7.3 Results
We evaluated collision counts (out of 50) for obstacles initialized at distances 3–7 m. Curriculum learning outperforms direct training at all distances, especially at 3 m.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/collisions_plot.png" title="Number of collisions at a given distance" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Number of collisions at each initialization distance (50 trials).
</div>

The quadruped learned the jump but often jumped too early at larger radii, landing before the obstacle arrived (Figure 6). Future work should encourage timed jumps.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/jump.png" title="Quadruped jumping over a flying obstacle" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
The quadruped clears a flying obstacle after training.
</div>

---

## 8 Conclusion
We demonstrated that PPO plus a two‑stage curriculum and modified RSI enable a quadruped to jump over moving obstacles. Curriculum learning accelerates skill acquisition and improves obstacle avoidance. Modified RSI broadens training states, and domain randomization aids sim‑to‑real transfer. Future directions include:
- Teaching the agent to jump *only* when collision is imminent.  
- Multiple successive jumps (jump‑roping).  
- Expert trajectories (true RSI) or imitation learning (DAgger).  
- Predictive modules for obstacle trajectories.

---
