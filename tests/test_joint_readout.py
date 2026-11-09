import unittest
import numpy as np
import torch
from training.joint_benchmarks import stratified
from training.joint_evaluate import field_views, ridge
from training.joint_data import JointData, image_patches


class JointReadoutTest(unittest.TestCase):
    def test_valid_ordered_bins_and_padding(self):
        v = torch.arange(20).reshape(2,10,1).float()
        lens = torch.tensor([2,7])
        views = field_views(v,lens)
        self.assertEqual(views["ordered"].shape,(2,4))
        self.assertEqual(views["last"].tolist(),[[1],[16]])
        v[0,2:] = 1e8
        torch.testing.assert_close(views["ordered"],field_views(v,lens)["ordered"])

    def test_exposure_cycle_and_heldout_rows(self):
        d = object.__new__(JointData)
        d.seed,d.batch = 81,64
        d.cfg = {"families":{"cifar":100}}
        order = np.concatenate([d.indices("cifar",i,67) for i in range(3)])
        self.assertEqual(len(set(order[:67])),67)
        self.assertEqual(len(set(order[67:134])),67)
        y = np.arange(400)%4
        fit,dev = (stratified(y,"fashion",r) for r in ("fit","dev"))
        self.assertFalse(set(fit)&set(dev))
        self.assertEqual(len(fit),128)

    def test_ridge_and_image_orientation(self):
        x = np.eye(4,dtype=np.float32)
        y = np.arange(4)
        prediction,_ = ridge(x,y,x,4,.1)
        np.testing.assert_array_equal(prediction,y)
        rgb = np.arange(3*32*32).reshape(1,3,32,32)
        patches = image_patches(rgb)
        recovered = patches.reshape(-1,4,4,3,8,8).transpose(0,3,1,4,2,5).reshape(rgb.shape)
        np.testing.assert_array_equal(recovered,rgb)
