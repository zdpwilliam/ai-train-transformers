import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoImageProcessor
from torchvision.transforms import RandomResizedCrop, ColorJitter, Compose


def transforms(examples):
    images = [_transforms(img.convert("RGB")) for img in examples["image"]]
    examples["pixel_values"] = image_processor(images, do_resize=False, return_tensors="pt")["pixel_values"]
    return examples


def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch


data_set = load_dataset("dataset/img/food101", split="train[:100]")
print(data_set[0]["image"])

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = Compose([RandomResizedCrop(size), ColorJitter(brightness=0.5, hue=0.5)])

data_set.set_transform(transforms)
img = data_set[0]["pixel_values"]
plt.imshow(img.permute(1, 2, 0))
plt.savefig("dataset/img/pretrain-img-after.png")
plt.show()