import json
import random
from pathlib import Path

# Check 1000-image subset
with open('Fiedler Evaluation/data/coco20i/coco20i_test_1000_random.json') as f:
    data_1000 = json.load(f)

print(f'COCO-20i 1000-random: {len(data_1000["images"])} images, {len(data_1000["annotations"])} annotations')

# Sample 500 from 1000
random.seed(42)
image_ids = [img['id'] for img in data_1000['images']]
selected_ids = set(random.sample(image_ids, 500))

selected_images = [img for img in data_1000['images'] if img['id'] in selected_ids]
selected_annotations = [ann for ann in data_1000['annotations'] if ann['image_id'] in selected_ids]

print(f'Sampled 500: {len(selected_images)} images, {len(selected_annotations)} annotations')

# Create and save
coco_500 = {
    'info': data_1000['info'],
    'licenses': data_1000['licenses'],
    'images': selected_images,
    'annotations': selected_annotations,
    'categories': data_1000['categories'],
}

with open('Fiedler Evaluation/data/coco_500_from_1000.json', 'w') as f:
    json.dump(coco_500, f)

print('Saved to: Fiedler Evaluation/data/coco_500_from_1000.json')
