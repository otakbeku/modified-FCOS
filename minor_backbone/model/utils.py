import funcy
import json
from pycocotools.coco import COCO
import numpy as np
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler
from torchvision import transforms
from PIL import Image
from torchvision.datasets import CocoDetection
import torch
import random
import cv2
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval


def get_train_val_test(annot_file, valid_size=0.2, test_size=0.2, with_randomsampler=False):
    print(annot_file)
    annotations_file = open(annot_file, 'rt', encoding='UTF-8')

    coco = json.load(annotations_file)
    images = coco['images']
    annotations = coco['annotations']
    coco = COCO(annot_file)

    images_with_annotations = funcy.lmap(
        lambda a: int(a['image_id']), annotations)

    images = funcy.lremove(
        lambda i: i['id'] not in images_with_annotations, images)

    dataset_size = len(images)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    test_split = int(np.floor(test_size * dataset_size))
    train_indices, test_indices = indices[test_split:], indices[:test_split]

    train_size = len(train_indices)
    valid_split = int(np.floor((1 - valid_size) * train_size))
    train_indices, valid_indices = train_indices[:
                                                 valid_split], train_indices[valid_split:]

    # Check for category in each splits
    for split in [train_indices, valid_indices, test_indices]:
        cats = [0] * 91

        for i in split:
            imageId = images[i]['id']
            annotationIds = coco.getAnnIds(imageId)
            annotations = coco.loadAnns(annotationIds)
            for i in range(len(annotations)):
                entityId = annotations[i]['category_id']
                cats[entityId] += 1
        print("training")
        print(cats[1:])
        l = 0
        for i in cats:
            if i == 0:
                l += 1
        print(l)

    annotations_file.close()
    train_ids = list(map(lambda x: images[x]['id'], train_indices))
    valid_ids = list(map(lambda x: images[x]['id'], valid_indices))
    test_ids = list(map(lambda x: images[x]['id'], test_indices))
    if with_randomsampler:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        return [train_ids, valid_ids, test_ids], [train_sampler, valid_sampler, test_sampler]
    else:
        return [train_ids, valid_ids, test_ids]

def model_dict_module_solver(pretrained_dict, model_dict):
    new_model_dict = model_dict.copy()
    for (pd, d) in tqdm(zip(pretrained_dict, model_dict)):
        new_model_dict[d] = pretrained_dict[pd]
    return new_model_dict

def flip(img, boxes):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    w = img.width
    if boxes.shape[0] != 0:
        xmin = w - boxes[:, 2]
        xmax = w - boxes[:, 0]
        boxes[:, 2] = xmax
        boxes[:, 0] = xmin
    return img, boxes

# buat training
class COCODataset(CocoDetection):
    CLASSES_NAME = (
        '__back_ground__', 'person', 'bicycle', 'car', 'motorcycle',
        'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
        'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet',
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush')

    def __init__(self, imgs_path, anno_path, sids, resize_size=[800, 1333], is_train=True, transform=None):
        super().__init__(imgs_path, anno_path)

        print("INFO====>check annos, filtering invalid data......")
        ids = []
        for id in sids:
            ann_id = self.coco.getAnnIds(imgIds=id, iscrowd=None)
            ann = self.coco.loadAnns(ann_id)
            if self._has_valid_annotation(ann):
                ids.append(id)
        self.ids = ids
        self.category2id = {v: i + 1 for i,
                            v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}

        self.transform = transform
        self.resize_size = resize_size

        self.mean = [0.40789654, 0.44719302, 0.47026115]
        self.std = [0.28863828, 0.27408164, 0.27809835]
        self.train = is_train

    def __getitem__(self, index):

        img, ann = super().__getitem__(index)

        ann = [o for o in ann if o['iscrowd'] == 0]
        boxes = [o['bbox'] for o in ann]
        boxes = np.array(boxes, dtype=np.float32)
        # xywh-->xyxy
        boxes[..., 2:] = boxes[..., 2:]+boxes[..., :2]
        if self.train:
            if random.random() < 0.5:
                img, boxes = flip(img, boxes)
            if self.transform is not None:
                img, boxes = self.transform(img, boxes)
        img = np.array(img)

        img, boxes = self.preprocess_img_boxes(img, boxes, self.resize_size)
        # img=draw_bboxes(img,boxes)

        classes = [o['category_id'] for o in ann]
        classes = [self.category2id[c] for c in classes]

        img = transforms.ToTensor()(img)
        # img= transforms.Normalize(self.mean, self.std,inplace=True)(img)
        boxes = torch.from_numpy(boxes)
        classes = torch.LongTensor(classes)

        return img, boxes, classes

    def preprocess_img_boxes(self, image, boxes, input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side = input_ksize
        h,  w, _ = image.shape

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side/smallest_side
        if largest_side*scale > max_side:
            scale = max_side/largest_side
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w = 32-nw % 32
        pad_h = 32-nh % 32

        image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes

    def _has_only_empty_bbox(self, annot):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)

    def _has_valid_annotation(self, annot):
        if len(annot) == 0:
            return False

        if self._has_only_empty_bbox(annot):
            return False

        return True

    def collate_fn(self, data):
        imgs_list, boxes_list, classes_list = zip(*data)
        assert len(imgs_list) == len(boxes_list) == len(classes_list)
        batch_size = len(boxes_list)
        pad_imgs_list = []
        pad_boxes_list = []
        pad_classes_list = []

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        for i in range(batch_size):
            img = imgs_list[i]
            pad_imgs_list.append(transforms.Normalize(self.mean, self.std, inplace=True)(
                torch.nn.functional.pad(img, (0, int(max_w-img.shape[2]), 0, int(max_h-img.shape[1])), value=0.)))

        max_num = 0
        for i in range(batch_size):
            n = boxes_list[i].shape[0]
            if n > max_num:
                max_num = n
        for i in range(batch_size):
            pad_boxes_list.append(torch.nn.functional.pad(
                boxes_list[i], (0, 0, 0, max_num-boxes_list[i].shape[0]), value=-1))
            pad_classes_list.append(torch.nn.functional.pad(
                classes_list[i], (0, max_num-classes_list[i].shape[0]), value=-1))

        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)
        batch_imgs = torch.stack(pad_imgs_list)

        return batch_imgs, batch_boxes, batch_classes

# Buat eval
class COCOGenerator(CocoDetection):
    CLASSES_NAME = (
        '__back_ground__', 'person', 'bicycle', 'car', 'motorcycle',
        'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
        'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet',
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush')

    def __init__(self, imgs_path, anno_path, sids, resize_size=[800, 1333]):
        super().__init__(imgs_path, anno_path)

        print("INFO====>check annos, filtering invalid data......")
        ids = []
        for id in sids:
            ann_id = self.coco.getAnnIds(imgIds=id, iscrowd=None)
            ann = self.coco.loadAnns(ann_id)
            if self._has_valid_annotation(ann):
                ids.append(id)
        self.ids = ids
        self.category2id = {v: i + 1 for i,
                            v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}

        self.resize_size = resize_size
        self.mean = [0.40789654, 0.44719302, 0.47026115]
        self.std = [0.28863828, 0.27408164, 0.27809835]

    def __getitem__(self, index):

        img, ann = super().__getitem__(index)

        ann = [o for o in ann if o['iscrowd'] == 0]
        boxes = [o['bbox'] for o in ann]
        boxes = np.array(boxes, dtype=np.float32)
        # xywh-->xyxy
        boxes[..., 2:] = boxes[..., 2:]+boxes[..., :2]
        img = np.array(img)

        img, boxes, scale = self.preprocess_img_boxes(
            img, boxes, self.resize_size)
        # img=draw_bboxes(img,boxes)

        classes = [o['category_id'] for o in ann]
        classes = [self.category2id[c] for c in classes]

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(self.mean, self.std, inplace=True)(img)
        # boxes=torch.from_numpy(boxes)
        classes = np.array(classes, dtype=np.int64)

        return img, boxes, classes, scale

    def preprocess_img_boxes(self, image, boxes, input_ksize):
        '''
        resize image and bboxes 
        Returns
        image_paded: input_ksize  
        bboxes: [None,4]
        '''
        min_side, max_side = input_ksize
        h,  w, _ = image.shape

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side/smallest_side
        if largest_side*scale > max_side:
            scale = max_side/largest_side
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w = 32-nw % 32
        pad_h = 32-nh % 32

        image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes, scale

    def _has_only_empty_bbox(self, annot):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)

    def _has_valid_annotation(self, annot):
        if len(annot) == 0:
            return False

        if self._has_only_empty_bbox(annot):
            return False

        return True


def evaluate_coco(generator, model, threshold=0.05):
    """ Use the pycocotools to evaluate a COCO model on a dataset.
    Args
        oU NMSgenerator : The generator for g
        model     : The model to evaluate.
        threshold : The score threshold to use.
    """
    # start collecting results
    results = []
    image_ids = []
    for index in tqdm(range(len(generator))):
        img, gt_boxes, gt_labels, scale = generator[index]
        # run network
        scores, labels, boxes = model(img.unsqueeze(dim=0).cuda())
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        boxes = boxes.detach().cpu().numpy()
        boxes /= scale
        # correct boxes for image scale
        # change to (x, y, w, h) (MS COCO standard)
        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]

        # compute predicted labels and scores
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted, so we can break
            if score < threshold:
                break

            # append detection for each positively labeled class
            image_result = {
                'image_id': generator.ids[index],
                'category_id': generator.id2category[label],
                'score': float(score),
                'bbox': box.tolist(),
            }

            # append detection to results
            results.append(image_result)

        # append image to list of processed images
        image_ids.append(generator.ids[index])

    if not len(results):
        return

    # write output
    json.dump(results, open('coco_bbox_results.json', 'w'), indent=4)
    # json.dump(image_ids, open('{}_processed_image_ids.json'.format(generator.set_name), 'w'), indent=4)

    # load results in COCO evaluation tool
    coco_true = generator.coco
    coco_pred = coco_true.loadRes('coco_bbox_results.json')

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats
