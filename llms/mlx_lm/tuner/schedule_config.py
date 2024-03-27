import mlx.optimizers as opt


def build_schedule(schedule_config):
    if schedule_config:
        schedule_name = schedule_config["name"]
        arguments = schedule_config["arguments"]
        initial_lr = arguments[0]
        schedule_fn = getattr(opt.schedulers, schedule_name)
        warmup_steps = schedule_config.get("warmup", 0)
        warmup_min_lr = schedule_config.get("minimum", 0.0)
        bound_schedule_fn = schedule_fn(*arguments)
        if warmup_steps:
            warmup_fn = opt.schedulers.linear_schedule(
                warmup_min_lr, initial_lr, warmup_steps
            )
            return opt.schedulers.join_schedules(
                [warmup_fn, bound_schedule_fn], [warmup_steps + 1]
            )
        else:
            return bound_schedule_fn
