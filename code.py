# !git clone https://github.com/fleur101/ORGAN

# !pip install -r ORGAN/requirements.txt

# !wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# !chmod +x Miniconda3-latest-Linux-x86_64.sh
# !time bash ./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local
# !time conda install -q -y -c conda-forge rdkit

# %matplotlib inline
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('/usr/local/lib/python3.7/site-packages/')

import sys
sys.path.append('/content/ORGAN/')
sys.path.append('/usr/local/lib/python3.7/site-packages/')
import organ
from organ import ORGAN

model = ORGAN('test1', 'mol_metrics', params={'PRETRAIN_DIS_EPOCHS': 1})
model.load_training_set('ORGAN/data/toy.csv')
model.set_training_program(['novelty', 'synthesizability'], [3, 1])
model.load_metrics()
model.train(ckpt_dir='checkpoints')

import tensorflow as tf
model_saver =  tf.train.Saver()
model_saver.save(model.sess, "test.ckpt")
# model.load_prev_training('test.ckpt')
results = model.generate_samples(20000)

from organ import mol_metrics

char_dict, ord_dict = mol_metrics.build_vocab()
smi = [mol_metrics.decode(i, ord_dict) for i in result]


file = open('smi.txt','w') 
file.write('\n'.join(smi))
file.close()



