# SpeakUp

SpeakUp is an audio-visual world model. Provided the context of what an agent has observed and heard (as well as actions it has previously taken), the model is able to generate several stochastic futures for both the video and audio. This all happens in latent space which allows for latent planning. Audio and Video are encoded into a shared latent space with pretrained encoders (DinoV3 and Dasheng). We use a transformer with flow matching to learn the distribution over future latens conditioned on past latents. Since this is inherently a continuous range of outcomes, sampling creates several feasible and disjoint futures compared to the mean of futures that regression-based approaches normally create. We then use a planner with CEM to search over sequences of actions, and this is cored against each sampled future. 

## Method

### Latents

Video runs at 10 fps. Each frame goes through a frozen DINOv3 ViT, giving a $16 \times 16$ grid of patch features per frame. Audio is loaded at 48 kHz:

- **Semantic audio:** a 16 kHz copy goes through a frozen audio encoder for each stereo channel, giving 5 tokens per 0.1 s step for Dasheng.
- **Spatial audio:** features computed from the raw 48 kHz waveform form a third audio channel. For stereo recordings these are interaural level and phase differences. For first-order ambisonics they are a sound-field descriptor.

At step $k$ the observation is

$$
z_k = \big(v_k \in \mathbb{R}^{P \times d_v},\; a_k \in \mathbb{R}^{d_a}\big),
$$

where $v_k$ holds the video patch tokens and $a_k$ the audio tokens. Latents are normalized per channel with statistics fitted on the training set. All latents are precomputed into a cacheso during training we never have to decode a video on the fly. The issue is that the actual size of these latents is VERY large, and optimizing this is currently still in progress.

### What we are actually trying to predict:

Provided $C$ context steps $z_{1:C}$ and actions $u_{1:K}$, the model generates the next $T$ steps $z_{C+1:C+T}$. For a video, it predicts the delta since the last context frame,

$$
x^{v}_{k} = v_{C+k} - v_{C},
$$

which removes the static scene from the target. Audio is predicted directly.

### Flow matching

Let $x_1$ be a target block (video changes plus audio) and $x_0 \sim \mathcal{N}(0, I)$ noise. The model learns the straight-line probability path between them:

$$
x_t = (1 - t)\,x_0 + t\,x_1, \qquad t \in [0, 1],
$$

whose velocity is constant along the path:

$$
\frac{d x_t}{d t} = x_1 - x_0 .
$$

A network $f_\theta$ predicts the clean sample $\hat x_1 = f_\theta(x_t, t, c)$, conditioned on context latents and actions $c$. That prediction is converted to a velocity:

$$
v_\theta(x_t, t, c) = \frac{\hat x_1 - x_t}{\max(1 - t,\ 0.05)} .
$$

The training loss is the flow-matching regression

$$
\mathcal{L}(\theta) = \mathbb{E}_{t,\,x_0,\,x_1,\,c}\Big[\big\| v_\theta(x_t, t, c) - (x_1 - x_0) \big\|^2\Big],
$$

summed over video and audio. To ensure that audio meaningfully contributes, we ablate setting audio weight in this loss to 0 for a "video-only" regime. 

Times are drawn as $t = \sigma(\xi)$ with $\xi \sim \mathcal{N}(0, 1)$ (logit-normal), then shifted toward the noisy end:

$$
t' = \frac{t}{s - (s - 1)\,t}, \qquad s = 3 .
$$

The choice of $s$ was ablated between 3,7,and 11. High-dimensional targets need more training signal at high noise, which is why the shift is there. 

### Sampling several futures

Generation is akin to an inverse problem, starting from pure noise and integrating the learned ODE with Euler steps on a shifted time grid $0 = t_0 < \dots < t_N = 1$:
$$
x_{t_{i+1}} = x_{t_i} + (t_{i+1} - t_i)\, v_\theta(x_{t_i}, t_i, c).
$$

Each noise draw $x_0$ gives a different future. Drawing $M$ samples gives an ensemble $\{\hat x^{(m)}\}_{m=1}^{M}$. We score it with the continuous ranked probability score,

$$
\mathrm{CRPS} = \frac{1}{M}\sum_{m} \big|\hat x^{(m)} - x\big| \;-\; \frac{1}{2M^2}\sum_{m,m'} \big|\hat x^{(m)} - \hat x^{(m')}\big| .
$$

It rewards an ensemble that is both accurate and appropriately spread. A deterministic model can only lower the first term. We report CRPS relative to copying the last context frame, so below 1 beats that baseline, and the spread-to-error ratio, where 1 is perfectly calibrated. We are also considering Heun sampling, but this is a later ablation over our euler solver

### Conditioning and robustness
Our main action dataset is D2E, which has recorded keyboard and mouse states. We embed and add this to the tokens of each step. With probability 0.1, actions are replaced by a learned null action, so the same model is also an action-free predictor. Ideally, this makes the model more generalizable as well. During training, context latents are corrupted as $\tilde z = (1-\sigma) z + \sigma \epsilon$ with $\sigma \sim \mathcal{U}(0, 0.2)$, and $\sigma$ is given to the model. This makes long rollouts tolerate the model's own imperfect predictions. Target tokens attend to target tokens at the same or earlier steps, and cross-attend to all context tokens.

### Architecture
Our world model is a double-stage LightningDiT style model. It uses AdaLN blocks conditioned on $t$ and the context-noise level, with cross-attention to the context tokens and a small, wide diffusion head that decodes the backbone features into per-token outputs, similar to how RAE approaches it. Video patch tokens and audio tokens are interleaved per time step.

### Planning

We use the forward model as an inverse model. Given true future latents as a goal, we run a cross-entropy-method search over action chunks. Each candidate is scored by the latent distance between the futures it produces and the goal. Inferred actions are compared with the ground truth actions associated with the window. Our eval measures how much predictions depend on actions: it compares true, shuffled and null actions and tests whether the true actions can be retrieved.

## Getting started

Everything uses [uv](https://docs.astral.sh/uv/).
The pyproject.toml we wrote has extras for different gpu hardware (or cpu), so pick the one which works best. Things like FA4 don't work below hopper, so pick carefully. 

### CPU baby test (synthetic data, should be good on a toaster, tested on thinkpad)
Sync cpu venv from pyproject.toml
```bash
uv sync --extra cpu
```

```bash
uv run --extra cpu -m data.latent_cache configs/pretrain/cpu.yaml --count 16
uv run --extra cpu -m data.latent_cache configs/pretrain/cpu.yaml --split validation
uv run --extra cpu -m training.pretrain configs/pretrain/cpu.yaml data.windows=16 trainer.max_steps=4
```

### Using actual datasets to train:

Each dataset is defined by a config in `configs/data/sources/<name>.yaml`:

```yaml
data:
  sources:
    d2e:
      root: ???                       # this is the path to the data!
      audio_root: ???
      splits: ${.root}/splits/d2e/{split}.txt
      video: ${.root}/media/{key}.mkv
      audio: ${.audio_root}/{key}.mka
      actions: ${.root}/media/{key}.mcap
```

Build the train latent cache, validation latent cache, train, then eval:

```bash
uv run -m data.latent_cache configs/pretrain/mvp.yaml
uv run -m data.latent_cache configs/pretrain/mvp.yaml --split validation
uv run -m training.pretrain configs/pretrain/mvp.yaml
uv run -m evals.evaluate <run_dir> <run_dir>/checkpoint-N
```

