from datasets.imagenet import get_data
from examples.imagenet.template import main
from models.architectures.resnet import get_resnet50_backbone

if __name__ == "__main__":
    nf = 64
    bs = 32

    resnet_backbone = get_resnet50_backbone(nf)

    train_batches, validation_batches = get_data(bs)

    main(resnet_backbone, "resnet50", train_batches, validation_batches)
