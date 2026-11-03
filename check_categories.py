import json

with open('Fiedler Evaluation/data/coco_500_from_1000.json') as f:
    data = json.load(f)

cat_names = {c['id']: c['name'] for c in data['categories']}
ann_cats = set(ann['category_id'] for ann in data['annotations'])
present_names = sorted(cat_names[c] for c in ann_cats)

print(f"Categories present in annotations: {len(present_names)}")
print(present_names)

coco20i_classes = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                    "boat", "dog", "cat", "bird", "horse", "sheep", "cow", "bottle",
                    "cup", "chair", "dining table", "backpack", "handbag"}
print(f"\nCOCO-20i target classes: {len(coco20i_classes)}")
extra = set(present_names) - coco20i_classes
print(f"Extra (non-COCO-20i) classes present: {len(extra)}")
print(sorted(extra))
