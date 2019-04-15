import organ
from organ import ORGAN

model = ORGAN('last_test', 'mol_metrics', params={'PRETRAIN_DIS_EPOCHS': 1})
model.load_training_set('data/zinc_1.csv')
model.set_training_program(['novelty', 'synthesizability'], [1, 1])
model.load_metrics()
model.train(ckpt_dir='checkpoints')

#import tensorflow as tf
#model_saver =  tf.train.Saver()
#model_saver.save(model.sess, "test.ckpt")
#model.load_prev_training('test.ckpt')
results = model.generate_samples(20000)

from organ import mol_metrics

char_dict, ord_dict = mol_metrics.build_vocab()
smi = [mol_metrics.decode(i, ord_dict) for i in results]


file = open('smi_organ.txt','w') 
file.write('\n'.join(smi))
file.close()



