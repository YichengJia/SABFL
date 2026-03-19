# Reproducibility and Implementation-Difference Report

- Generated at: 2026-03-08T13:43:40.388035Z
- Num clients profile: 50
- Strict baseline profiles: True

## 1) Protocol Configuration Suite

### fedavg

```json
{
  "strict_reproduction": true,
  "participation_rate": 1.0,
  "use_timeout": false,
  "max_round_time": 10.0
}
```

### fedasync

```json
{
  "max_staleness": 10,
  "learning_rate": 0.8,
  "staleness_mode": "linear",
  "staleness_floor": 0.1,
  "server_tick_sec": 0.02
}
```

### fedbuff

```json
{
  "buffer_size": 5,
  "max_staleness": 15,
  "server_lr": 0.2,
  "gradient_clip": 5.0
}
```

### scaffold

```json
{
  "strict_reproduction": true,
  "participation_rate": 1.0,
  "use_timeout": false,
  "learning_rate": 0.8,
  "max_round_time": 10.0
}
```

### improved_async_balanced

```json
{
  "max_staleness": 15,
  "min_buffer_size": 3,
  "max_buffer_size": 6,
  "momentum": 0.85,
  "adaptive_weighting": true,
  "auto_scale_params": true,
  "staleness_quantile": 0.9,
  "staleness_history_size": 512,
  "min_effective_staleness": 5,
  "max_effective_staleness": 128,
  "server_lr": 0.2,
  "gradient_clip": 2.0
}
```

### improved_async_low_comm_topk

```json
{
  "max_staleness": 20,
  "min_buffer_size": 5,
  "max_buffer_size": 10,
  "momentum": 0.8,
  "adaptive_weighting": true,
  "compression": "topk",
  "k": 100,
  "auto_scale_params": true,
  "staleness_quantile": 0.9,
  "staleness_history_size": 512,
  "min_effective_staleness": 5,
  "max_effective_staleness": 128,
  "server_lr": 0.2,
  "gradient_clip": 2.0
}
```

### improved_async_balanced_qsgd

```json
{
  "max_staleness": 15,
  "min_buffer_size": 3,
  "max_buffer_size": 6,
  "momentum": 0.85,
  "adaptive_weighting": true,
  "compression": "qsgd",
  "num_bits": 8,
  "auto_scale_params": true,
  "staleness_quantile": 0.9,
  "staleness_history_size": 512,
  "min_effective_staleness": 5,
  "max_effective_staleness": 128,
  "server_lr": 0.2,
  "gradient_clip": 2.0
}
```

## 2) Implementation Intent and Known Deviations

- **fedavg**
  - intent: Synchronous fixed-size rounds under strict mode.
  - known_deviation: No paper-specific optimizer/loss schedule beyond common training loop.
- **fedasync**
  - intent: Queue-driven async server updates with staleness decay.
  - known_deviation: Server scheduler is generic tick-based, not tied to a specific paper runtime stack.
- **fedbuff**
  - intent: Buffered asynchronous aggregation with stale filtering.
  - known_deviation: Server LR and clipping remain configurable engineering knobs.
- **scaffold**
  - intent: Control-variate based sync rounds with strict gating.
  - known_deviation: Control variate update is simplified for unified framework compatibility.
- **improved_async**
  - intent: Main method: scale-aware, staleness-aware, adaptive-buffer async FL.
  - known_deviation: A research extension, not a direct reproduction target.

## 3) Suggested Paper Language

- Baselines are implemented under strict profile settings in a unified codebase.
- We report implementation differences explicitly and avoid claiming bit-level original-code reproduction.
- Main contribution is ImprovedAsync with scale-aware bounded controls and system-observed staleness adaptation.
