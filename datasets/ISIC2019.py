import os.path as osp

from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class ISIC2019(DatasetBase):

    dataset_dir = "ISIC2019"
    domains = ["BCN", "HAM"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        self.open_set = cfg.DATASET.OPEN_SET if hasattr(cfg.DATASET, "OPEN_SET") else False
        self.known_classes = cfg.DATASET.KNOWN_CLASSES if hasattr(cfg.DATASET, "KNOWN_CLASSES") else None
        self.unknown_classes = cfg.DATASET.UNKNOWN_CLASSES if hasattr(cfg.DATASET, "UNKNOWN_CLASSES") else None

        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS, is_source=True)
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS, is_source=False)
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, is_source=False)

        super().__init__(train_x=train_x, train_u=train_u, test=test)

    def _read_data(self, input_domains, is_source=True):
        items = []

        for domain, dname in enumerate(input_domains):
            domain_dir = osp.join(self.dataset_dir, dname)
            class_names = listdir_nohidden(domain_dir)
            class_names.sort()
            

            if self.open_set:
                if is_source and self.known_classes is not None:

                    class_names = [c for c in class_names if c in self.known_classes]
                elif not is_source and self.unknown_classes is not None:

                    pass

            for label, class_name in enumerate(class_names):
                class_path = osp.join(domain_dir, class_name)
                imnames = listdir_nohidden(class_path)
                

                if self.open_set and not is_source and self.unknown_classes is not None and class_name in self.unknown_classes:

                    if self.known_classes is not None:
                        actual_label = len(self.known_classes)  
                    else:
                        actual_label = -1 
                else:

                    if self.open_set and self.known_classes is not None:
                        actual_label = self.known_classes.index(class_name) if class_name in self.known_classes else -1
                    else:
                        actual_label = label

                for imname in imnames:
                    impath = osp.join(class_path, imname)
                    item = Datum(
                        impath=impath,
                        label=actual_label,
                        domain=domain,
                        classname=class_name
                    )
                    items.append(item)

        return items
