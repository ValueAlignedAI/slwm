import unittest
import torch
from training.joint_structured import prune


class StructuredWeightsTest(unittest.TestCase):
    def test_magnitude_pattern_and_stable_ties_preserve_parent(self):
        w = torch.tensor([[1.,-3.,2.,0., 1.,-1.,1.,-1.],
                          [0.,0.,0.,0., .1,-.2,.3,-.4]])
        before = w.clone()
        actual = prune(w)
        expected = torch.tensor([[0.,-3.,2.,0., 1.,-1.,0.,0.],
                                 [0.,0.,0.,0., 0.,0.,.3,-.4]])
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(w,before)
        self.assertEqual(actual.shape,w.shape)


if __name__=="__main__":
    unittest.main()
