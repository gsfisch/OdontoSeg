import timm

all_models = [m for m in timm.list_models() if 'pvt' in m]

print(all_models)
