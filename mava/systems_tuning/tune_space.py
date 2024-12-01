from carbs import CARBSParams, LogitSpace, LogSpace, Param

param_spaces = [
    Param(name="actor_lr", space=LogSpace(scale=0.5, min=1e-6, max=1e-3), search_center=1e-4),
    Param(name="clip_eps", space=LogitSpace(min=0.01, max=0.7), search_center=0.2),
    Param(name="ppo_epochs", space=LogSpace(is_integer=True, min=1, max=10), search_center=4),
]
carbs_params = CARBSParams(
    better_direction_sign=1,
    is_wandb_logging_enabled=False,
    resample_frequency=0,
)
