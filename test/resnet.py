from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from datasets import load_dataset

class CustomModel(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
        self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        x = self.feature_extractor(x)  # 提取特征
        x = self.flatten(x.pooler_output)  # 分类
        return x

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]
model = CustomModel()

image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")


inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(inputs['pixel_values'])

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
