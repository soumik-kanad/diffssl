FEAT_DIM_LIST = [256]*7 + [512]*6 + [1024]*12 + [512]*6 + [256]*6
FEAT_SIZE_LIST = [256]*3 + [128]*3 + [64]*3 + [32]*3 + [16]*3+ [8]*6 + [16]*3 + [32]*3 + [64]*3 + [128]*3 + [256]*4

DM_FEAT_DIM_DICT = {}
DM_FEAT_SIZE_DICT = {}
for idx, val in enumerate(FEAT_DIM_LIST):
    DM_FEAT_DIM_DICT[idx+1] = val

for idx, val in enumerate(FEAT_SIZE_LIST):
    DM_FEAT_SIZE_DICT[idx+1] = val

RESNET_FEAT_DIM_DICT = {
    1: 256,
    2: 512,
    3: 1024,
    4: 2048,
}

RESNET_FEAT_SIZE_DICT = {
    1: 56,
    2: 28,
    3: 14,
    4: 7,
}

VIT_FEAT_DIM_DICT = {
    i: 768 for i in range(1, 13)
}