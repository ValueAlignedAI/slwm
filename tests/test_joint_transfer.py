import unittest
import torch
from training.t1_torch_text import TorchGPT2LM
from models.joint_transfer import JointWorld, SharedUpdate, objective


class JointTransferTest(unittest.TestCase):
    def world(self, experts):
        parent = TorchGPT2LM(vocab_size=101,context_length=48,n_layer=2,n_embd=32,n_head=4)
        return JointWorld(parent,{"rank":8,"experts":experts},81)

    def test_causality_and_parent_identity(self):
        x = torch.randint(0,101,(3,12))
        for experts in (1,4):
            m = self.world(experts).eval()
            state,_,counts = m.field({"text":x})
            torch.testing.assert_close(state@m.core.token_embedding.weight.T,m.core(x))
            for count in counts:
                self.assertEqual(int(count.sum()),x.numel())
            with torch.no_grad():
                for layer in m.updates:
                    for e in layer.experts:
                        e.up.weight.normal_()
            y = x.clone()
            y[:,6:] = (y[:,6:]+1)%101
            torch.testing.assert_close(m.field({"text":x})[0][:,:6],m.field({"text":y})[0][:,:6])
            original = m.field({"text":x})[0]
            clone = self.world(experts)
            clone.core.load_state_dict(m.core.state_dict())
            clone.apply_delta(m.delta())
            torch.testing.assert_close(original,clone.field({"text":x})[0])

    def test_signal_gradients_and_absence(self):
        m = self.world(4)
        values = torch.randn(3,9,64)
        mask = torch.rand_like(values)>.2
        batch = {"numeric":values, "present":mask, "image":torch.randn(3,9,192),
                 "caption":torch.randint(0,101,(3,5)),"lengths":torch.tensor([5,4,0]),
                 "caption_keys":torch.arange(3),"record_keys":torch.arange(3)}
        loss, _, _ = objective(m,batch)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in m.core.parameters()))
        self.assertGreater(m.edges["numeric"].weight.grad.abs().sum().item(),0)
        self.assertGreater(m.align.weight.grad.abs().sum().item(),0)
        a = m.field(batch)[0]
        torch.testing.assert_close(a,m.field({**batch,"numeric":values.masked_fill(~mask,1234)})[0])
        torch.testing.assert_close(a,m.field({**batch,"caption":torch.zeros_like(batch["caption"])})[0])


if __name__=="__main__":
    unittest.main()
