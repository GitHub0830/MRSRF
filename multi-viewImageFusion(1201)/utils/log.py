##################################################
#borrowed from https://github.com/nashory/pggan-pytorch
##################################################
import torch
import numpy as np
import torchvision.models as models
import utils.utils as utils
import os, sys
from datetime import datetime as dt
# from tensorboardX import SummaryWriter


class TensorBoardX:
    def __init__(self, config_filename, code_filename, output_path, sub_dir ="" ):
        os.system('mkdir -p {}/{}'.format(output_path, sub_dir))
        self.path = '{}/{}/{}'.format(output_path, sub_dir , dt.now().isoformat())
        
        if not os.path.exists(self.path):
            print("Saving logs at {}".format(self.path))
            # self.writer = {}
            # self.writer['train'] = SummaryWriter(self.path+'/train')
            # self.writer['val'] = SummaryWriter( self.path + '/val' )
            os.system('mkdir -p {}'.format(self.path) )
            os.system('cp {} {}/'.format(config_filename , self.path))
            os.system('cp {} {}/'.format(code_filename , self.path))


             
    def add_scalar(self, index, val, niter , logtype):
        self.writer[logtype].add_scalar(index, val, niter)


    def add_scalars(self, index, group_dict, niter , logtype):
        self.writer[logtype].add_scalar(index, group_dict, niter)

    def add_image_grid(self, index, ngrid, x, niter , logtype):
        grid = utils.make_image_grid(x, ngrid)
        self.writer[logtype].add_image(index, grid, niter)

    def add_image_single(self, index, x, niter , logtype):
        self.writer[logtype].add_image(index, x, niter)

    def add_histogram(self, index , x , niter , logtype):
        self.writer[logtype].add_histogram( index , x , niter )


    def add_graph(self, index, x_input, model , logtype):
        torch.onnx.export(model, x_input, os.path.join(self.path, "{}.proto".format(index)), verbose=True)
        self.writer[logtype].add_graph_onnx(os.path.join(self.path, "{}.proto".format(index)))

    def export_json(self, out_file , logtype ):
        self.writer[logtype].export_scalars_to_json(out_file)

