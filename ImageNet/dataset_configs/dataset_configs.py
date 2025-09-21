args_dcrt = {
    'subset0': {
        'attack': 'mi_fgsm',
        'eps': 4/255,
        'source_model': [
            {
            'arch': 'resnet50',
            }
        ],
        'label_smoothing': 1.0,
    },
    'subset1': {
        'attack': 'di_fgsm',
        'eps': 4/255,
        'source_model': [
            {
            'arch': 'resnet50',
            }
        ],
        'label_smoothing': 1.0,
    },
    'subset2': {
        'attack': 'fia',
        'eps': 4/255,
        'source_model': [
            {
            'arch': 'resnet50',
            }
        ],
        'label_smoothing': 1.0,
    },
    'subset3': {
        'attack': 'IAA',
        'eps': 4/255,
        'source_model': [
            {
            'arch': 'resnet50',
            }
        ],
        'label_smoothing': 1.0,
    },
    'subset4': {
        'attack': 'GAP',
        'eps': 4/255,
        'source_model': [
            {
            'arch': 'resnet50',
            }
        ],
        'label_smoothing': 1.0,
    },
}