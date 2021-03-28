import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.gramMatrix import StyleLoss
import torchvision
import torch.nn.functional as F
from util.wassestein_loss import calc_gradient_penalty


class WarpingClothModel(BaseModel):
    def name(self):
        return 'WarpingClothModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['L1', 'gan', 'G', 'D']
        # specify the images G_A'you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_image_mask', 'cloth_mask', 'warped_cloth']

        self.visual_names = visual_names_A
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        use_sigmoid = opt.no_lsgan
        self.netD = networks.define_D(3, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_image = input['base_image'].to(self.device)
        self.real_image_mask = input['base_image_mask'].to(self.device)
        self.real_cloth = input['base_cloth'].to(self.device)
        self.real_cloth_mask = input['base_cloth_mask'].to(self.device)

    def forward(self):
        self.image_mask = self.real_image.mul(self.real_image_mask)
        self.cloth_mask = self.real_cloth.mul(self.real_cloth_mask)

        #cloth warping fake
        self.warped_cloth = self.netG(torch.cat([self.real_image_mask, self.cloth_mask], dim=1))

    def backward_G(self):
        self.loss_L1 = 10 * self.criterionL1(self.warped_cloth, self.image_mask)
        self.loss_gan = self.criterionGAN(self.netD(self.warped_cloth), True)
        self.loss_G = self.loss_L1 + self.loss_gan
        self.loss_G.backward(retain_graph=True)

    def backward_D(self):
        # pred_real = self.netD(self.image_mask)
        # self.loss_D_1 = self.criterionGAN(pred_real, True)
        #
        # pred_fake = self.netD(self.warped_cloth.detach())
        # self.loss_D_2 = self.criterionGAN(pred_fake, False)
        # self.loss_D = (self.loss_D_1 + self.loss_D_2) * 0.5
        grad_penalty_A = calc_gradient_penalty(self.netD, self.warped_cloth, self.image_mask)
        self.loss_D = torch.mean(self.netD(self.warped_cloth)) - torch.mean(self.netD(self.image_mask)) + grad_penalty_A
        self.loss_D.backward(retain_graph=True)

    def optimize_parameters(self):
        # forward
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()


