from .model.fcos import FCOSDetector
import torch
from .model.utils import COCODataset
from .model.coco_eval import COCOGenerator
import math,time
from .dataset.augment import Transforms
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler
from .model.utils import get_train_val_test

class Options():
    #backbone
    epochs = 50
    batch_size = 4
    n_cpu = 4
    n_gpu = '0'
    valid_size = 0.2
    test_size = 0.2

# kaggle
COCO_TRAIN = ".../input/coco-2017-dataset/coco2017/train2017/"
COCO_VAL = "../input/coco-2017-dataset/coco2017/val2017/"
COCO_ANNOT = "../input/coco-2017-dataset/coco2017/annotations/"

opt = Options
os.environ["CUDA_VISIBLE_DEVICES"]=opt.n_gpu
BATCH_SIZE=opt.batch_size
EPOCHS=opt.epochs
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
transform = Transforms()
(train_ids, valid_ids, test_ids) = get_train_val_test(COCO_ANNOT + 'instances_val2017.json')
# Dataset
train_dataset=COCODataset(COCO_VAL,
                          COCO_ANNOT + 'instances_val2017.json',transform=transform, sids=train_ids)
valid_dataset=COCODataset(COCO_VAL,
                          COCO_ANNOT + 'instances_val2017.json',transform=transform, sids=valid_ids)
test_dataset=COCODataset(COCO_VAL,
                          COCO_ANNOT + 'instances_val2017.json',transform=transform, sids=test_ids)

# Generator
train_generator = COCOGenerator(COCO_VAL,
                          COCO_ANNOT + 'instances_val2017.json', sids=train_ids)
valid_generator = COCOGenerator(COCO_VAL,
                          COCO_ANNOT + 'instances_val2017.json', sids=valid_ids)
test_generator = COCOGenerator(COCO_VAL,
                          COCO_ANNOT + 'instances_val2017.json', sids=test_ids)

# DataLoader
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=train_dataset.collate_fn,
                                         num_workers=opt.n_cpu,worker_init_fn = np.random.seed(0))

valid_loader=torch.utils.data.DataLoader(valid_dataset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=valid_dataset.collate_fn,
                                         num_workers=opt.n_cpu,worker_init_fn = np.random.seed(0))

test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=test_dataset.collate_fn,
                                         num_workers=opt.n_cpu,worker_init_fn = np.random.seed(0))
print(f'Train: {len(train_loader)}\nValidation: {len(valid_loader)}\nTest: {len(test_loader)}')

model=FCOSDetector(mode="training", backbone='res2net50_48w_2s').cuda()
model = torch.nn.DataParallel(model)

steps_per_epoch=len(train_dataset)//BATCH_SIZE
TOTAL_STEPS=steps_per_epoch*EPOCHS
WARMUP_STEPS=500
WARMUP_FACTOR = 1.0 / 3.0
GLOBAL_STEPS=0
LR_INIT=0.01
optimizer = torch.optim.SGD(model.parameters(),lr =LR_INIT,momentum=0.9,weight_decay=0.0001)
lr_schedule = [120000, 160000]
def lr_func(step):
    lr = LR_INIT
    if step < WARMUP_STEPS:
        alpha = float(step) / WARMUP_STEPS
        warmup_factor = WARMUP_FACTOR * (1.0 - alpha) + alpha
        lr = lr*warmup_factor
    else:
        for i in range(len(lr_schedule)):
            if step < lr_schedule[i]:
                break
            lr *= 0.1
    return float(lr)


# train
data = next(iter(train_loader))

batch_imgs,batch_boxes,batch_classes=data
batch_imgs=batch_imgs.cuda()
batch_boxes=batch_boxes.cuda()
batch_classes=batch_classes.cuda()

lr = lr_func(GLOBAL_STEPS)
for param in optimizer.param_groups:
    param['lr']=lr

start_time=time.time()

optimizer.zero_grad()
losses=model([batch_imgs,batch_boxes,batch_classes])
loss=losses[-1]
loss.mean().backward()
torch.nn.utils.clip_grad_norm(model.parameters(),3)
optimizer.step()

end_time=time.time()
cost_time=int((end_time-start_time)*1000)
print("Train global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f"%\
        (GLOBAL_STEPS,0+1,0+1,steps_per_epoch,losses[0].mean(),losses[1].mean(),losses[2].mean(),cost_time,lr, loss.mean()))


GLOBAL_STEPS+=1
torch.cuda.synchronize()
print()


torch.cuda.empty_cache() # This should be safe, but might negatively affect the performance, since PyTorch might need to reallocate this memory again.