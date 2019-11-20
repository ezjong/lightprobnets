from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import zipfile
from utils import system
import os


def create_zip(filename, directory, include_extensions=('*.py', '*.sh')):
    filenames = []
    arcdir = os.path.basename(filename.split('.')[0])
    for ext in include_extensions:
        filenames += system.get_filenames(directory, match=ext)
    with zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED) as file:
        for f in filenames:
            arcname = f.replace(directory, arcdir)
            file.write(f, arcname)
