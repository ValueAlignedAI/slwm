"""Make a byte-reproducible inference file from one exact optimizer checkpoint and its calibrated priors."""
import os
from pathlib import Path
import shutil
import zipfile


def export(resume, prior, destination):
    tmp=destination.with_suffix(".inference-part")
    with zipfile.ZipFile(resume) as state,zipfile.ZipFile(prior) as cal,zipfile.ZipFile(tmp,"w",allowZip64=True) as output:
        names=sorted(n for n in state.namelist() if n.startswith("p/") and n.endswith(".npy"))
        if not names:
            raise ValueError("No model weights in optimizer checkpoint")
        members=[(state,n,n[2:]) for n in names]+[(state,"stats.npy","stats.npy"),
                 (cal,"coef.npy","prior_coef.npy"),(cal,"contact.npy","prior_contact.npy")]
        for source,member,name in members:
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.create_system=3
            info.external_attr=0o600<<16
            with source.open(member) as reader,output.open(info,"w",force_zip64=True) as writer:
                shutil.copyfileobj(reader,writer,4*1024**2)
    os.replace(tmp,destination)
