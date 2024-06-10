from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class CustomMountedEmpty(BaseSegDataset):

    METAINFO = dict(
        classes=[],
        palette=(),
    )

    def __init__(
        self,
        target_class_map=None,
        classes: list = None,
        img_suffix: str = '.jpg',
        seg_map_suffix: str = '.png',
        **kwargs
    ) -> None:
        print(f"Found classes: {classes}")
        self.METAINFO["classes"] = []
        for cls in classes:
            if cls in target_class_map:
                if target_class_map[cls] is not None:
                    self.METAINFO["classes"].append(target_class_map[cls])
            else:
                self.METAINFO["classes"].append(cls)

        print(f"Classes for training: {self.METAINFO['classes']}")
        print(f"Target_class_map: {target_class_map}")

        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

