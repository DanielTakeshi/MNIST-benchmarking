"""
Put file names here so that we can load easily into plotting.  Arrange oldest at
bottom, most recent at the top.
(c) 2018 by Daniel Seita
"""

ATTRIBUTES = [
    'ValidAvgAcc',   'ValidAvgDiff',
    'TestAvgAcc',    'TestAvgDiff',
    'RegressLoss',   'L2RegLoss',
    'Img1AvgValL2',  'Img2AvgValL2',
    'Img1AvgTestL2', 'Img2AvgTestL2',
]

# ------------------------------------------------------------------------------
# Exp 02: to see 3k steps, again still in exploration stage ...
# ------------------------------------------------------------------------------

s1='train-2600-epochs-3000-bsize-200-arch-2-lrate-0.001-seed'
s2='train-2600-epochs-3000-bsize-200-arch-2-lrate-0.0001-seed'
MAR12_3000_EPOCHS = {
    'directories': [
            ['{}-{}'.format(s1,b) for b in range(0,3)],
            ['{}-{}'.format(s2,b) for b in range(0,3)],
    ],
    'names': [
            'tr-2600-eps-3000-bs-200-arch-2-lr-0.001',
            'tr-2600-eps-3000-bs-200-arch-2-lr-0.0001',
    ],
}

# ------------------------------------------------------------------------------
# Exp 01: first real trial with just 2600 data points.
# ------------------------------------------------------------------------------

s1='train-2600-epochs-1000-bsize-200-arch-1-lrate-0.001-seed'
s2='train-2600-epochs-1000-bsize-200-arch-2-lrate-0.001-seed'
s3='train-2600-epochs-1000-bsize-200-arch-2-lrate-0.0001-seed'
MAR07_1000_EPOCHS = {
    'directories': [
            ['{}-{}'.format(s1,b) for b in range(0,4)],
            ['{}-{}'.format(s2,b) for b in range(0,4)],
            ['{}-{}'.format(s3,b) for b in range(0,4)],
    ],
    'names': [
            'tr-2600-eps-1000-bs-200-arch-1-lr-0.001',
            'tr-2600-eps-1000-bs-200-arch-2-lr-0.001',
            'tr-2600-eps-1000-bs-200-arch-2-lr-0.0001',
    ],
}


s1='train-2600-epochs-100-bsize-200-arch-1-bnorm-False-seed'
s2='train-2600-epochs-100-bsize-200-arch-2-bnorm-False-seed'
MAR07_FIRST_TRY = {
    'directories': [
            ['{}-{}'.format(s1,b) for b in range(0,4)],
            ['{}-{}'.format(s2,b) for b in range(0,4)],
    ],
    'names': [
            'tr-2600-eps-100-bs-200-arch-1',
            'tr-2600-eps-100-bs-200-arch-2',
    ],
}
