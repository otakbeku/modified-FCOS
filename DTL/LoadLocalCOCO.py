import json
import cv2
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO


class LoadLocalCOCO:
    def __init__(self, file_path, coco_root, category_names=['dog', 'cat']):
        with open(file_path, 'rb') as f:
            self.data = json.load(f)
        self.supercategories = set(x['supercategory']
                                   for x in self.data['categories'])
        self.categories = {x['name']: x['id'] for x in self.data['categories']}
        self.id_categories = {x['id']: x['name']
                              for x in self.data['categories']}
        self.image_df = pd.DataFrame(self.data['images'])
        self.annotation_df = pd.DataFrame(self.data['annotations'])
        self.root_folder = Path(coco_root)
        self.coco = COCO(file_path)
        self.excluded_image = set(self.coco.getCatIds(catNms=category_names))

    def get_category_id(self, category_name):
        return self.categories[category_name.lower()]

    def _get_image_by_id(self, image_id):
        image_path = self.image_df[self.image_df.file_name.str.contains(
            str(image_id))].file_name.values[0]
        image = cv2.imread(str(self.root_folder/image_path))
        return image

    def get_cropped_images(self, category_name) -> list:
        idx_cat = self.get_category_id(category_name.lower())
        images = []
        image_ids = self.annotation_df[self.annotation_df.category_id == idx_cat][[
            'image_id', 'bbox']]
        for frames in tqdm(image_ids.itertuples()):
            if frames.image_id in self.excluded_image:
                continue
            image = self._get_image_by_id(frames.image_id)
            image = self._crop_image(image, frames.bbox)
            images.append(image)
        return images

    def get_cropped_images_flatten_and_resized(self, category_name, resize=(150, 150), flatten=True) -> list:
        idx_cat = self.get_category_id(category_name.lower())
        images = []
        image_ids = self.annotation_df[self.annotation_df.category_id == idx_cat][[
            'image_id', 'bbox']]
        broken_ids = []
        for frames in tqdm(image_ids.itertuples()):
            if frames.image_id in self.excluded_image:
                continue
            image = self._get_image_by_id(frames.image_id)
            image = self._crop_image(image, frames.bbox)
            try:
                image = cv2.resize(image, resize)
                if flatten:
                    image = image.flatten()
                images.append(image)
            except Exception as identifier:
                broken_ids.append(frames.image_id)
                continue
        print(f'Broken images: {broken_ids} images')
        return images

    def get_number_of_images_per_category(self, category_name):
        idx_cat = self.get_category_id(category_name.lower())
        return len(self.annotation_df[self.annotation_df.category_id == idx_cat].index)

    @staticmethod
    def _crop_image(image, bbox):
        x, y, w, h = list(map(int, bbox))
        return image[y:y+h, x:x+w]
