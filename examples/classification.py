from tensorflow.keras import Sequential

from models.architectures.resnet import get_resnet50_backbone, ClassificationHead


class Classificator(Sequential):
    def __init__(self, backbone, head):
        super().__init__([
            backbone,
            head
        ])


if __name__ == "__main__":
    class_model = Classificator(
        backbone=get_resnet50_backbone(64),
        head=ClassificationHead(10))
