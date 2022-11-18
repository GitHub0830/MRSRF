OUT = {}
OUT['path'] = "/home/jjq/paper/ckpts_P40pro/"

#***************************** for HUAWEI***************************#
DATA = {}
##dataset

# DATA['train_Folder'] = "/home/disk60t/jjq/imageFusion/20211031/datas/train/"
# DATA['val_Folder'] = "/home/disk60t/jjq/imageFusion/20211031/datas/val/"
# DATA['test_Folder'] = "/home/disk60t/jjq/imageFusion/20211031/datas/test/" 

DATA['train_Folder'] = "/home/jjq/paper/data/x4/train/"
DATA['val_Folder'] = "/home/jjq/paper/data/x4/val/"
DATA['test_Folder'] = "/home/jjq/paper/data/x4/test/"

##trainning
DATA['batch_size'] = 4
DATA['val_batch_size'] = 1 
DATA['train_num_workers'] = 8
DATA['val_num_workers'] = 4
DATA['test_batch_size'] = 1 
DATA['test_num_workers'] = 4

##optimizer
#optimizer
DATA['learning_rate'] = 1e-4
DATA['lr_decay'] = 0.1
DATA['lr_milestone'] = [80,160,240]
DATA['momentum'] = 0.9
DATA['beta'] = 0.999
#model
DATA['leaky_value'] = 0.1
DATA['num_epochs'] = 400 #400
DATA['log_epoch'] = 10 
DATA['resume'] = None  
DATA['resume_epoch'] = None  
DATA['resume_optimizer'] = './save'

