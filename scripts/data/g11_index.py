"""Read immutable public HoloAssist member offsets for physical-effect measurement."""
import argparse
from pathlib import Path
import g10_holo_index as tar_index

PHYSICAL = {
    "ahat_depth": ('"0x8DC8C9E2FD16A8F"', 601785251840),
    "hands": ('"0x8DC8C9DE150FE51"', 235404984320),
    "eyes": ('"0x8DC8C9D71756078"', 2634506240),
    "cam_info": ('"0x8DC8C9DFE3DEF24"', 10815559680),
}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("archive",choices=PHYSICAL)
    a=p.parse_args();tar_index.ARCHIVES.update(PHYSICAL);tar_index.index(a.root,a.archive)
