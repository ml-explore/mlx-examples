import unittest
import yaml
import math
import mlx.optimizers as opt
from mlx_lm.schedule_config import build_schedule
from mlx_lm.lora import yaml_loader


CONFIG_YAML1 = """
schedule:
  join:
    boundaries: [101]
    schedules:
      - { name: linear_schedule,
          arguments: [0.0, 1e-5, 100] }
      - { name: cosine_decay,
          arguments: [ 1e-5, 100 ] }
"""


class TestScheduleConfigs(unittest.TestCase):
    def test_join(self):
        config = yaml.load(CONFIG_YAML1, yaml_loader)
        cos_with_warmup = build_schedule(config['schedule'])
        self.assertIsNotNone(cos_with_warmup)

        self.assertEqual(cos_with_warmup(0), 0.0)
        self.assertAlmostEqual(cos_with_warmup(101), 1e-5, delta=1e-1)
        optimizer = opt.Adam(learning_rate=cos_with_warmup)
        for _ in range(100):
            optimizer.update({}, {})
        self.assertAlmostEqual(optimizer.learning_rate.item(), 1e-5, delta=1e-1)
        for _ in range(100):
            optimizer.update({}, {})
        expected_lr = 1e-5 * 0.5 * (1.0 + math.cos(math.pi * 200 / 10))
        self.assertAlmostEqual(optimizer.learning_rate.item(), expected_lr, delta=1e-1)
