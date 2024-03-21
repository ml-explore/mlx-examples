import mlx.optimizers as opt


def build_schedule(schedule_config):
    if schedule_config:
        if schedule_config and "join" in schedule_config:
            join_config = schedule_config["join"]
            boundaries = join_config["boundaries"]
            schedules = []
            for schedule in join_config["schedules"]:
                schedule_name = schedule["name"]
                arguments = schedule["arguments"]
                schedule_fn = getattr(opt.schedulers, schedule_name)
                schedules.append(schedule_fn(*arguments))
            return opt.schedulers.join_schedules(schedules, boundaries)
        else:
            for schedule_name, options in schedule_config.items():
                arguments = options["arguments"]
                schedule_fn = getattr(opt.schedulers, schedule_name)
                return schedule_fn(*arguments)
