
args_dcrt = {
    'subset0': {
        'attack': 'mi_fgsm',
        'eps': 8/255,
        'source_model': [
            {
            'arch': 'resnet50',
            'ckpt_path': '/data/crq/ckpt/dcrt/resnet50_TinyImagenet.pth',
            },
        ],
        'merge_mode': 'sampling',
        'label_smoothing': 1.0,
    },
    'subset1': {
        'attack': 'di_fgsm',
        'eps': 8/255,
        'source_model': [
            {
            'arch': 'resnet50',
            'ckpt_path': '/data/crq/ckpt/dcrt/resnet50_TinyImagenet.pth',
            },
        ],
        'merge_mode': 'sampling',
        'label_smoothing': 1.0,
    },
    'subset2': {
        'attack': 'fia',
        'eps': 8/255,
        'source_model': [
            {
            'arch': 'resnet50',
            'ckpt_path': '/data/crq/ckpt/dcrt/resnet50_TinyImagenet.pth',
            },
        ],
        'merge_mode': 'sampling',
        'label_smoothing': 1.0,
    },
    'subset3': {
        'attack': 'IAA',
        'eps': 8/255,
        'source_model': [
            {
            'arch': 'resnet50',
            'ckpt_path': '/data/crq/ckpt/dcrt/resnet50_TinyImagenet.pth',
            },
        ],
        'merge_mode': 'sampling',
        'label_smoothing': 1.0,
    },
    'subset4': {
        'attack': 'GAP',
        'eps': 8/255,
        'source_model': [
            {
            'arch': 'resnet50',
            'ckpt_path': '/data/crq/ckpt/dcrt/resnet50_TinyImagenet.pth',
            },
        ],
        'merge_mode': 'sampling',
        'label_smoothing': 1.0,
    },
}