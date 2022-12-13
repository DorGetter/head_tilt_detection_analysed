# -*- coding: utf-8 -*-
"""
@author: dorge
"""

# =============================================================================
# file operations lib 
# =============================================================================
#     filname:        file_operation.py
#     version:        1.0
#     author:         dorge
#     creation_date:  11-12-2022
#     
#     change history:
#     
#     who       when      version     changes
#     ----      -----     -------     --------
#     dorge     11-12-22   1.0         INIT
#
#   description: used to provide all file related functionality.
# =============================================================================
import time
import numpy as np
from typing import List

from apps.core.logger import Logger


def create_file(file_path: str):
    with open(f'{file_path}', 'w') as f:
        f.close()
    pass


def append_detections_to_text_file(filepath: str,
                                   single_frame_dets: List[np.ndarray]):
    with open(filepath, 'a') as f:
        f.write(f"{time.time()}\t")
        f.write(f"{[list(d) for d in single_frame_dets]}\n")
    time.sleep(0.1)  # e.g. send to a remote server


def rows(f, chunksize=256, sep='\n'):
    """
    generator: 
    Read a file where the row separator is '\n'.
    """
    row = ''
    while (chunk := f.read(chunksize)) != '':   # End of file
        while (i := chunk.find(sep)) != -1:     # No separator found
            yield row + chunk[:i]
            chunk = chunk[i+1:]
            row = ''
        row += chunk
    yield row
