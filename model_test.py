import torch
import unittest
from torch.utils.data import DataLoader
from model import RawNet2
from dataset import ASVspoofDataset
import yaml

class TestRawNet2(unittest.TestCase):
    def setUp(self):
        # Initialize model and dataset
        config_path = '/home/ubuntu/anti_spoof/config.yaml'
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.model = RawNet2(**cfg['model'])
        self.sample_input = torch.randn(1, 1, 64000)  # Adjust the shape according to your model's input

    def test_forward_pass(self):
        """ Test forward pass of the model. """
        try:
            output = self.model(self.sample_input)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Forward pass raised an exception: {e}")

    def test_output_shape(self):
        """ Test the output shape of the model. """
        output = self.model(self.sample_input)
        expected_shape = (1, 2)  # Adjust according to your expected output shape
        self.assertEqual(output.shape, expected_shape, f"Output shape is not as expected: {output.shape}")

if __name__ == '__main__':
    unittest.main()
