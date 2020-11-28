from model.fcos import FCOSDetector
import torch
from dataset.COCO_dataset import COCODataset
import math,time
from dataset.augment import Transforms
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler

class Options():
    #backbone
    epochs = 50
    batch_size = 4
    n_cpu = 4
    n_gpu = '0'
    valid_size = 0.2
    test_size = 0.2

# # kaggle
# COCO_TRAIN = ".../input/coco-2017-dataset/coco2017/train2017/"
# COCO_VAL = "../input/coco-2017-dataset/coco2017/val2017/"
# COCO_ANNOT = "../input/coco-2017-dataset/coco2017/annotations/"

# local
COCO_TRAIN = ".../input/coco-2017-dataset/coco2017/train2017/"
COCO_VAL = "D:/FSR/COCO/val2017/"
COCO_ANNOT = "D:/FSR/COCO/annotations_trainval2017/annotations/"


opt = Options
os.environ["CUDA_VISIBLE_DEVICES"]=opt.n_gpu
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
transform = Transforms()
dataset=COCODataset(COCO_VAL,
                          COCO_ANNOT + 'instances_val2017.json',transform=transform)

dataset_size = len(dataset)
# indices = list(torch.randperm((dataset_size)))
indices = list(range(dataset_size))
np.random.shuffle(indices)
# print(indices)

test_split = int(np.floor(opt.test_size * dataset_size))
train_indices, test_indices = indices[test_split:], indices[:test_split]

train_size = len(train_indices)
valid_split = int(np.floor((1 - opt.valid_size) * train_size))
train_indices, valid_indices = train_indices[:valid_split], train_indices[valid_split:]

# print(len(train_indices), len(valid_sampler), len(test_sampler))

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)
test_sampler = SubsetRandomSampler(test_indices)

# pin memory buat training aja, terus empty cache
# nggak disarankan disatu operasi
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    sampler=train_sampler,
    collate_fn=dataset.collate_fn,
    num_workers=opt.n_cpu,
    worker_init_fn = np.random.seed(0), 
#     pin_memory=True
)

valid_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    sampler=valid_sampler,
    collate_fn=dataset.collate_fn,
    num_workers=opt.n_cpu,
    worker_init_fn = np.random.seed(0), 
#     pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    sampler=test_sampler,
    collate_fn=dataset.collate_fn,
    num_workers=opt.n_cpu,
    worker_init_fn = np.random.seed(0), 
#     pin_memory=True
)

model=FCOSDetector(mode="training", backbone='res2net50_48w_2s').cuda()
model = torch.nn.DataParallel(model)
BATCH_SIZE=opt.batch_size
EPOCHS=opt.epochs

steps_per_epoch=len(dataset)//BATCH_SIZE
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
    
# Train
model.train()
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