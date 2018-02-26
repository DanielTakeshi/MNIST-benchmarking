""" 
Put plot names here if too unwiedly to put in other code.  Do it in reverse
chronological order.
"""
# Feb 26, RMSProp, coarse.
FEB26_NOVALID = {
    'hparams': {
        'lrate': ['0.01', '0.005', '0.001', '0.0005'],
        'wd':    ['0.0', '0.000001', '0.00001', '0.0001'],
    },
    'logdir': 'logs/rmsprop-novalid-tune/',
    'figdir': 'figures/tune_rmsprop_coarse_novalid.png',
}

## # Feb 25, RMSProp, coarse.
## hparams = {
##     'lrate': ['0.01', '0.005', '0.001', '0.0005', '0.0001'],
##     'wd':    ['0.0', '0.000001', '0.00001', '0.0001'],
## }
## plot_one_type('logs/rmsprop-tune/', "figures/tune_rmsprop_coarse.png", hparams)
## 
## # Feb 25, RMSProp, fine. These have 20 random seeds.
## hparams = {
##     'lrate': ['0.001', '0.002', '0.003', '0.004', '0.005'],
##     'wd':    ['0.000001', '0.00001'],
## }
## plot_one_type('logs/rmsprop-fine-tune/', "figures/tune_rmsprop_fine.png", hparams)
