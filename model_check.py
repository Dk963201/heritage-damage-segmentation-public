"""Synthetic software smoke test only: NOT training or research evaluation."""
import json, time
from pathlib import Path
import torch
import segmentation_models_pytorch as smp


def build_model(name):
    constructors={'unet':smp.Unet,'deeplabv3plus':smp.DeepLabV3Plus,'fpn':smp.FPN}
    return constructors[name](encoder_name='efficientnet-b0',encoder_weights=None,in_channels=3,classes=4)


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    device=torch.device('cpu')
    records=[]
    for name in ['unet','deeplabv3plus','fpn']:
        start=time.monotonic()
        model=build_model(name).to(device).train()
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        warmup=torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=0.1,total_iters=5)
        x=torch.rand(2,3,64,64,device=device)
        y=torch.randint(0,4,(2,64,64),device=device)
        logits=model(x)
        assert logits.shape==(2,4,64,64)
        loss=smp.losses.FocalLoss(mode='multiclass')(logits,y)+smp.losses.DiceLoss(mode='multiclass')(logits,y)
        assert torch.isfinite(loss)
        loss.backward();optimizer.step();warmup.step()
        records.append(dict(model=name,encoder='efficientnet-b0',output_shape=list(logits.shape),
            finite_loss=True,backward_and_optimizer_step=True,seconds=round(time.monotonic()-start,2)))
        print(name,'PASS',flush=True)
        del model, optimizer, warmup, logits, loss
    output=dict(test_type='SYNTHETIC SOFTWARE CHECK ONLY',device=str(device),
        mps_available=torch.backends.mps.is_available(),torch_version=torch.__version__,
        smp_version=smp.__version__,models=records,trained_on_heritage_images=False)
    Path('model-smoke-results.json').write_text(json.dumps(output,indent=2)+'\n')

if __name__=='__main__':main()
