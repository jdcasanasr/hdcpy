import numpy as np

from hdcpy               import *
from hdcpy_auxiliary        import *
from hdcpy_classification   import *

signal = np.random.uniform(-1.0, 1.0, 617)

level_item_memory   = get_level_hypermatrix(10, 10000, 'MAP')
id_item_memory      = get_id_hypermatrix(617, 10000, 'MAP')

encoded_signal      = encode_signal(signal, level_item_memory, id_item_memory, 'MAP')

pass