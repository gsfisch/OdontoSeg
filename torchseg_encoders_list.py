import torchseg

encoders = [x for x in torchseg.list_encoders() if 'swin' in x and 'base' in x]
print(encoders)
