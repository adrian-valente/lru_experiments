import unittest
import torch
import model

class TestLRU(unittest.TestCase):
    
    def test_init(self):
        net = model.LRU(1, 4, 1)
        net = model.LRU(2, 3, 4)
    
    def test_forward(self):
        net = model.LRU(1, 4, 1)
        inputs = torch.Tensor([[0., 0, 1, 1, 0]]).unsqueeze(2)
        y = net(inputs)
        assert y.shape == inputs.shape
        assert y.dtype == inputs.dtype
    
    def test_batch(self):
        net = model.LRU(1, 4, 1)
        inputs = torch.Tensor([[0., 0, 1, 1, 0], [1., 1, 0, 0, 0]]).unsqueeze(2)
        y = net(inputs)
        assert y.shape == inputs.shape
        assert y.dtype == inputs.dtype
        
    def test_init_states(self):
        net = model.LRU(1, 4, 1)
        inputs = torch.Tensor([[0., 0, 1, 1, 0], [1., 1, 0, 0, 0]]).unsqueeze(2)
        
        # Batch init states
        init_states = torch.Tensor([[1., 1, 1, 1], [1., 1, 1, 1]])
        y = net(inputs, init_states)
        
        # Non-batch init states
        init_states = torch.Tensor([[1., 1, 1, 1]])
        y = net(inputs, init_states)


class TestSequenceModel(unittest.TestCase):
    
    def test_init(self):
        net = model.SequenceLayer(1, 4, [1, 2, 1])
        net = model.SequenceLayer(2, 3, [4, 5, 6])
    
    def test_forward(self):
        net = model.SequenceLayer(1, 4, [1, 2, 1])
        inputs = torch.Tensor([[0., 0, 1, 1, 0], [1., 1, 0, 0, 0]]).unsqueeze(2)
        y = net(inputs)
        assert y.shape == inputs.shape
        assert y.dtype == inputs.dtype
        
    def test_nlin(self):
        net = model.SequenceLayer(1, 4, [1, 2, 1], non_linearity=torch.nn.functional.gelu)
        inputs = torch.Tensor([[0., 0, 1, 1, 0], [1., 1, 0, 0, 0]]).unsqueeze(2)
        y = net(inputs)
        assert y.shape == inputs.shape
        assert y.dtype == inputs.dtype
        
    def test_skip(self):
        net = model.SequenceLayer(1, 4, [1, 2, 1], skip_connection=True)
        inputs = torch.Tensor([[0., 0, 1, 1, 0], [1., 1, 0, 0, 0]]).unsqueeze(2)
        y = net(inputs)
        assert y.shape == inputs.shape
        assert y.dtype == inputs.dtype
        
        
class TestDeepLRUModel(unittest.TestCase):
    
    def test_init(self):
        net = model.DeepLRUModel(1, 4, 3, [1, 2, 1], [3, 10])
        net = model.DeepLRUModel(2, 3, 1, [4, 5, 6])
        
    def test_forward(self):
        net = model.DeepLRUModel(1, 4, 3, [1, 2, 1], [3, 10])
        inputs = torch.Tensor([[0., 0, 1, 1, 0], [1., 1, 0, 0, 0]]).unsqueeze(2)
        y = net(inputs)
        assert y.shape == (inputs.shape[0], inputs.shape[1], 10)
        assert y.dtype == inputs.dtype
        
if __name__ == '__main__':
    unittest.main()