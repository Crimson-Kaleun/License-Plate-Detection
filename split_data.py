import os
import random
import shutil
from pathlib import Path

# Пути
all_images = Path("Nums_Work")
dataset_dir = Path("data/dataset")

# Создаем папки
for split in ['train', 'val']:
    (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
    (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

# Получаем все файлы
images = list(all_images.glob("*.jpg")) + list(all_images.glob("*.png"))
print(f"Images:::::")
print(f"All: {len(images)} images")
random.shuffle(images)

# 80% на обучение, 20% на проверку
split_idx = int(len(images) * 0.8)
train_images = images[:split_idx]
val_images = images[split_idx:]

# Копируем файлы
train_count = 0
val_count = 0
for img in train_images:
    # Копируем соответствующий txt файл
    txt_file = img.with_suffix('.txt')
    if txt_file.exists():
        shutil.copy(txt_file, dataset_dir / 'train' / 'labels' / txt_file.name)
        # Копируем картинку
        shutil.copy(img, dataset_dir / 'train' / 'images' / img.name)
        train_count += 1

for img in val_images:
    txt_file = img.with_suffix('.txt')
    if txt_file.exists():
        shutil.copy(txt_file, dataset_dir / 'val' / 'labels' / txt_file.name)
        shutil.copy(img, dataset_dir / 'val' / 'images' / img.name)
        val_count += 1

# print(f"Train: {len(train_images)} images")
# print(f"Val: {len(val_images)} images")
print(f"Train: {train_count} images")
print(f"Val: {val_count} images")