from mmengine.config import Config


def _set_dataset_classes(ds, target_class_map: dict, classes: list):
    if ds.get('type') == 'ConcatDataset':
        for sub in ds['datasets']:
            _set_dataset_classes(sub, target_class_map, classes)
    else:
        ds['target_class_map'] = target_class_map
        ds['classes'] = classes


def make_mmseg_config(config_path: str, target_class_map: dict, classes: list) -> Config:
    """Setup classes in custom_vit_uper"""
    cfg = Config.fromfile(config_path)

    # Model config
    num_classes = len(classes) + 1
    cfg.model.auxiliary_head.num_classes = num_classes
    cfg.model.auxiliary_head.out_channels = num_classes
    cfg.model.decode_head.num_classes = num_classes
    cfg.model.decode_head.out_channels = num_classes

    # Datasets config
    _set_dataset_classes(cfg.train_dataloader.dataset, target_class_map, classes)
    _set_dataset_classes(cfg.test_dataloader.dataset, target_class_map, classes)
    _set_dataset_classes(cfg.val_dataloader.dataset, target_class_map, classes)

    return cfg