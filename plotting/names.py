"""
Put file names here so that we can load easily into plotting.  Arrange oldest at
bottom, most recent at the top.
(c) 2018 by Daniel Seita
"""

ATTRIBUTES = [
    'ValidAvgAcc', 'ValidAvgDiff',
    'TestAvgAcc',  'TestAvgDiff',
    'RegressLoss', 'L2RegLoss',
    'Img1ValidL2', 'Img2ValidL2',
    'Img1TestL2',  'Img2TestL2',
]

s1='train-55000-epocs-50-bsize-100-arch-1-bnorm-False-seed'
s2='train-55000-epocs-50-bsize-100-arch-2-bnorm-False-seed'
MAR06_NUMTRAIN_55000 = {
    'directories': [
            ['{}-{}'.format(s1,b) for b in range(0,4)],
            ['{}-{}'.format(s2,b) for b in range(0,4)],
    ],
    'names': [
            'tr-55000-eps-50-bs-100-arch-1', 
            'tr-55000-eps-50-bs-100-arch-2',
    ],
}
