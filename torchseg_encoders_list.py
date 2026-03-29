import torchseg

encoders = [x for x in torchseg.list_encoders()]
print(encoders)
