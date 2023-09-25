import unittest
import torch
import model
import train

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
        net = model.DeepLRUModel(1, 4, 3, [1, 2, 1], [10])
        net = model.DeepLRUModel(2, 3, 1, [4, 5, 6])
        
    def test_forward(self):
        net = model.DeepLRUModel(1, 4, 3, [1, 2, 2], [10])
        inputs = torch.Tensor([[0., 0, 1, 1, 0], [1., 1, 0, 0, 0]]).unsqueeze(2)
        y = net(inputs)
        assert y.shape == (inputs.shape[0], inputs.shape[1], 10)
        assert y.dtype == inputs.dtype


class TestDSModel(unittest.TestCase):
    
    def test_init(self):
        net = model.DSModel(3, 1, 128, 2, [56, 56], [256, 256], output_widths=[3], lru_kwargs={"skip_connection": True})
        net = model.DSModel(3, 1, 128, 2, [56, 56], [256, 256], lru_kwargs={"skip_connection": True})
    
    def test_forward(self):
        ds_dim = 3
        net = model.DSModel(ds_dim, 1, 128, 2, [56, 56], [256, 256], output_widths=[ds_dim], lru_kwargs={"skip_connection": True})
        inputs = torch.zeros(32, 40, 1)
        init_states = torch.randn(32, ds_dim)
        y = net(inputs, init_states)
        assert y.shape == (32, 40, ds_dim)
        assert y.dtype == inputs.dtype
        
        
class TestTrain(unittest.TestCase):
    
    def test_train(self):
        net = model.DeepLRUModel(1, 10, 1, [10, 10], [1])
        x_train = torch.zeros(32, 100, 1)
        y_train = torch.cos(0.1 * torch.arange(100)).tile(32, 1, 1)
        x_test = x_train.clone()
        y_test = y_train.clone()
        train.train(net, x_train, y_train, x_test, y_test, n_epochs=1, lr=1e-3)
        
    def test_cuda(self):
        if not torch.cuda.is_available():
            print("No CUDA device available, didn't test CUDA training.")
            return 1
        net = model.DeepLRUModel(1, 10, 1, [10, 10], [1])
        x_train = torch.zeros(32, 100, 1)
        y_train = torch.cos(0.1 * torch.arange(100)).tile(32, 1, 1)
        x_test = x_train.clone()
        y_test = y_train.clone()
        train.train(net, x_train, y_train, x_test, y_test, n_epochs=1, lr=1e-3, cuda=True)
        print("Tested CUDA.")

        
if __name__ == '__main__':
    unittest.main()