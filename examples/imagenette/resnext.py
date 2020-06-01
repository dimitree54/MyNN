from examples.imagenette.template import main
from models.architectures.resnext import get_resnext50_backbone

if __name__ == "__main__":
    nf = 64
    bs = 64
    resnext_backbone = get_resnext50_backbone(nf)
    main(resnext_backbone, "resnext50", bs)
