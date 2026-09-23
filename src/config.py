
from datasets_mgr.babylm import use_babylm
from datasets_mgr.finewebedu import use_fineweb_edu

from datasets_mgr import unified_dataset_loader

DATASET_LOADER = unified_dataset_loader([use_babylm(), use_fineweb_edu()])

