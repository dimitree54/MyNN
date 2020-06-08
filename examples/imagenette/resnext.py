from datasets.imagenette import get_data
from examples.imagenette.template import main
from models.architectures.resnext import get_resnext50_backbone

if __name__ == "__main__":
    nf = 64
    bs = 32
    resnext_backbone = get_resnext50_backbone(nf)

    train_batches, validation_batches = get_data(bs)

    main(resnext_backbone, "resnext50", train_batches, validation_batches)
