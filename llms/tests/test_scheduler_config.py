import math
import unittest

import mlx.optimizers as opt
import yaml
from mlx_lm.lora import yaml_loader
from mlx_lm.schedule_config import build_schedule

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

CONFIG_YAML2 = """
schedule:
  join:
    schedules:
      - { name: linear_schedule,
          arguments: [0.0, 1e-5, 100] }
      - { name: cosine_decay,
          arguments: [ 1e-5, 100 ] }
"""

CONFIG_YAML3 = """
schedule:
  join:
"""

CONFIG_YAML4 = """
schedule:
  cosine_decay:
    arguments: [ 0.1, 10 ]
"""

CONFIG_YAML5 = """
schedule:
  
"""


class TestScheduleConfigs(unittest.TestCase):
    def test_join(self):
        config = yaml.load(CONFIG_YAML1, yaml_loader)
        cos_with_warmup = build_schedule(config["schedule"])
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

    def test_single_schedule(self):
        config = yaml.load(CONFIG_YAML4, yaml_loader)
        lr_schedule = build_schedule(config["schedule"])
        lr = lr_schedule(4)
        expected_lr = 0.1 * 0.5 * (1.0 + math.cos(math.pi * 4 / 10))
        self.assertAlmostEqual(lr, expected_lr, delta=1e-7)

    def test_malformed_config(self):
        config = yaml.load(CONFIG_YAML2, yaml_loader)
        self.assertRaises(KeyError, build_schedule, config["schedule"])

        config = yaml.load(CONFIG_YAML3, yaml_loader)
        self.assertRaises(TypeError, build_schedule, config["schedule"])

    def test_empty_config(self):
        config = yaml.load(CONFIG_YAML5, yaml_loader)
        self.assertIsNone(build_schedule(config["schedule"]))


if __name__ == "__main__":
    unittest.main()
