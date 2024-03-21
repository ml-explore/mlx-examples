import mlx.optimizers as opt


def build_schedule(config):
    schedule_config = config['schedule']
    if 'join' in schedule_config:
        boundaries = schedule_config['boundaries']
        schedules = []
        for schedule in schedule_config['schedules']:
            schedule_name = schedule['name']
            arguments = schedule['arguments']
            schedule_fn = getattr(opt.schedulers, schedule_name)
            schedules.append(schedule_fn(*arguments))
        return opt.schedulers.join_schedules(schedules, boundaries)