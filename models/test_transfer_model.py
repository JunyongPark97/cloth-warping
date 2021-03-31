from .base_model import BaseModel
from .warping_cloth_transfer_model import WarpingClothTransfermodel
from . import networks
import torch
import os


class TestTransferModel(WarpingClothTransfermodel):
    def name(self):
        return 'TestTransferModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestTrnasferModel cannot be used in train mode'
        parser.set_defaults(dataset_mode='sgunit_test')
        parser.add_argument('--model_suffix', type=str, default='_A',
                            help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will'
                            ' be loaded as the generator of TestTransferModel')

        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['image_mask', 'input_mask', 'warped_cloth', 'fake_image', 'final_image']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G' + opt.model_suffix]

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_warp = networks.define_G(opt.input_nc_warp, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                           opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_warp.module.load_state_dict(
            torch.load(os.path.join("./checkpoints/warping_model", 'latest_net_G_warp.pth')))

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see BaseModel.load_networks
        setattr(self, 'netG' + opt.model_suffix, self.netG)

    def set_input(self, input):
        self.real_image = input['base_image'].to(self.device)
        self.real_image_mask = input['base_image_mask'].to(self.device)
        self.real_cloth = input['base_cloth'].to(self.device)
        self.real_cloth_mask = input['base_cloth_mask'].to(self.device)
        self.input_cloth = input['input_cloth'].to(self.device)
        self.input_cloth_mask = input['input_cloth_mask'].to(self.device)

    def forward(self):
        self.image_mask = self.real_image.mul(self.real_image_mask)
        self.cloth_mask = self.real_cloth.mul(self.real_cloth_mask)
        self.input_mask = self.input_cloth.mul(self.input_cloth_mask)
        self.warped_cloth = self.netG_warp(torch.cat([self.real_image_mask, self.input_mask], dim=1))

        self.fake_image = self.netG_A(torch.cat([self.warped_cloth, self.image_mask], dim=1))

        self.fake_image = self.fake_image.mul(self.real_image_mask)

        self.empty_image = torch.sub(self.real_image, self.image_mask)
        self.final_image = torch.add(self.empty_image, self.fake_image)
