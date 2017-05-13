from matplotlib import pyplot as plt
import numpy as np

from path import *
from signdet import *

numcol = 4
fig = plt.figure(figsize=(5.5, 5.5))
sns = sorted(signs.keys())
for i, sn in enumerate(sns):
    img = plt.imread(signs[sn])
    print(np.ceil(len(signs)*1.0/numcol), numcol, i+1)
    plt.subplot(np.ceil(len(signs)*1.0/numcol), numcol, i+1)
    plt.imshow(img)
    plt.axis('off')
    sn = sn.replace('pedestrian_crossing', 'pedestrian')
    sn = sn.replace('parking_area', 'parking')
    plt.title(sn, fontsize=10)
# plt.tight_layout()
# plt.show()
plt.savefig('{0}/signs.pdf'.format(SCRATCH_PATH), dpi=fig.dpi)
