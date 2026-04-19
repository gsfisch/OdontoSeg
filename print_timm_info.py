import timm
import torchseg

# Encoders in TorchSeg
encoders = [x for x in torchseg.list_encoders()]
print(encoders)

# Models in timm
all_models = [m for m in timm.list_models()]
print(all_models)

# Metadata
metadata = torchseg.encoders.TIMM_ENCODERS["vit_tiny_patch16_224"]
print(metadata)