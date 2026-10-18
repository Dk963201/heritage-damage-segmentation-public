import tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from train_four_class import loss_fn,starts,evaluate,Patches
class FourClassTests(unittest.TestCase):
    def test_ignore_has_no_gradient(self):
        x=torch.randn(1,4,2,2,requires_grad=True);y=torch.tensor([[[0,1],[2,255]]])
        loss_fn(x,y).backward();self.assertEqual(float(x.grad[0,:,1,1].abs().sum()),0)
    def test_all_ignored_finite_zero(self):
        x=torch.randn(1,4,2,2,requires_grad=True)
        loss=loss_fn(x,torch.full((1,2,2),255));loss.backward()
        self.assertEqual(loss.item(),0);self.assertEqual(x.grad.abs().sum().item(),0)
    def test_tiles_cover_edges(self):
        for n in [15,32,33,63,64,65,101]:
            covered=set()
            for x in starts(n,32):covered.update(range(x,min(x+32,n)))
            self.assertEqual(covered,set(range(n)))
    def test_full_image_counts_once_and_ignores(self):
        class Healthy(torch.nn.Module):
            def forward(self,x):
                out=torch.zeros(x.shape[0],4,*x.shape[2:]);out[:,0]=1;return out
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);Image.new('RGB',(35,37)).save(root/'im.png')
            mask=np.zeros((37,35),np.uint8);mask[0,:]=255;mask[1,:]=1
            Image.fromarray(mask).save(root/'mask.png')
            rows=[{'image_path':'im.png','mask_path':'mask.png','site':' Amedi ','building_group':'fixture'}]
            r=evaluate(Healthy(),rows,root,32,False)
            self.assertEqual(sum(map(sum,r['confusion_matrix'])),36*35)
            self.assertEqual(r['confusion_matrix'][1][0],35)
            self.assertEqual(r['by_site']['Amedi']['confusion_matrix'],r['confusion_matrix'])
            self.assertEqual(r['class_support'],[35*35,35,0,0])
            self.assertIsNone(r['class_iou'][2])
            self.assertEqual(r['per_image'][0]['building_group'],'fixture')
            both=evaluate(Healthy(),rows+[dict(rows[0],site='Erbil')],root,32,False)
            pooled=np.array(both['by_site']['Amedi']['confusion_matrix'])+np.array(both['by_site']['Erbil']['confusion_matrix'])
            self.assertEqual(pooled.tolist(),both['confusion_matrix'])
    def test_small_image_padding_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);Image.new('RGB',(10,12)).save(root/'i.png');Image.new('L',(10,12),2).save(root/'m.png')
            ds=Patches([{'image_path':'i.png','mask_path':'m.png'}],root,32,False)
            _,mask=ds[0];self.assertEqual(int((mask!=255).sum()),120)
if __name__=='__main__':unittest.main()
