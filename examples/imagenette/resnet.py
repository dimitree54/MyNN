from datasets.imagenette import get_data
from examples.imagenette.template import main
from models.architectures.resnet import get_resnet50_backbone

if __name__ == "__main__":
    nf = 64
    bs = 64

    resnet_backbone = get_resnet50_backbone(nf)

    train_batches, validation_batches = get_data(bs)

    main(resnet_backbone, "resnet50", train_batches, validation_batches)
