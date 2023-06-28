import argparse
import math
import random
import os
import sys
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import viz
from copy import deepcopy
import numpy

class RandomFlip(object):
    """Horizontally flip the given numpy Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (numpy array): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            img = np.flip(img,axis=0)
        if random.random() < self.p:
            img = np.flip(img,axis=1)
        return np.ascontiguousarray(img)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomRotate(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        
        return np.ascontiguousarray(np.rot90(img,random.randint(0,3)))

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

try:
    import wandb

except ImportError:
    wandb = None


from model import Generator, Extra
from model import Patch_Discriminator as Discriminator  # , Projection_head
from model import Patch_Patch_Discriminator 
from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from non_leaking import augment
#use matplotlib as substitution
from matplotlib import pyplot as plt

#need to be rewritten
def save_image(tensor, latent, fp, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0, format=None,use_wandb = False, iteration = 0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp - A filename(string) or file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    #print(tensor[2,0,...].squeeze())
    grid = utils.make_grid(tensor[:,0,...].unsqueeze(1), nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.to('cpu', torch.float32).numpy()
    #print(ndarr[1])
    latent=latent.to('cpu', torch.float32).numpy()
    plt.figure(figsize=(10,14))
    plt.imshow(ndarr[0],vmin=0,vmax=3*np.std(ndarr[0]))
    y_ticks=[]
    y_str=[]
    for i in [0,1,2,3]:
        y_ticks.append(128+i*258)
        y_str.append('{:.3f}'.format(latent[i][0]*2+4)+','+'{:.1f}'.format(10**(latent[i][1]*1.398+1)))
    plt.yticks(y_ticks,y_str)
    plt.colorbar()
    plt.savefig(fp+'.png', format=format,dpi=300,bbox_inches='tight')
    if use_wandb:
        wandb.log({'tb':wandb.Image(fp+'.png')},step=iteration)
    plt.close()

    grid = utils.make_grid(tensor[:,1,...].unsqueeze(1), nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.to('cpu', torch.float32).numpy()
    #latent=latent.to('cpu', torch.float32).numpy()
    plt.figure(figsize=(10,14))
    plt.imshow((ndarr[0]+1)*2-1)
    y_ticks=[]
    y_str=[]
    for i in [0,1,2,3]:
        y_ticks.append(128+i*258)
        y_str.append('{:.3f}'.format(latent[i][0]*2+4)+','+'{:.1f}'.format(10**(latent[i][1]*1.398+1)))
    plt.yticks(y_ticks,y_str)
    plt.colorbar()
    plt.savefig(fp+'_rho.png', format=format,dpi=300,bbox_inches='tight')
    if use_wandb:
        wandb.log({'mass':wandb.Image(fp+'_rho.png')},step=iteration)
    plt.close()

def init_generator(generator):
    state_dic = generator.state_dict()
    input_tensor = state_dic['input.input']
    state_dic.update({'input.input2':input_tensor.clone()})
    state_dic.update({'input.input3':input_tensor.clone()})
    state_dic.update({'input.input4':input_tensor.clone()})
    generator.load_state_dict(state_dic)

def init_discriminator(discriminator):
    state_dic = discriminator.state_dict()
    input_tensor = state_dic['pre_final_linear.0.weight']
    new_tensor = torch.cat((input_tensor,input_tensor,input_tensor,input_tensor),axis = 1)
    #new_dic={}
    state_dic.update({'pre_final_linear_1.0.weight':new_tensor})
    input_tensor = state_dic['pre_final_linear.0.bias']
    state_dic.update({'pre_final_linear_1.0.bias':input_tensor.clone()})
    discriminator.load_state_dict(state_dic)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for name, p in model.named_parameters():
        p.requires_grad = flag

def requires_named_grad(model,keywords, flag=False):
    for name, p in model.named_parameters():
        for keyword in keywords:
            if keyword in name:
                p.requires_grad=flag

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(
        grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * \
        (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def make_label(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.rand(batch, latent_dim, device=device)

    noises = torch.rand(n_noise, batch, latent_dim, device=device)

    return noises


def mixing_noise(batch,param_dim, latent_dim, prob, device):
    label = make_label(batch,param_dim,1,device)
    if prob > 0 and random.random() < prob:
        return label,make_noise(batch, latent_dim, 2, device)

    else:
        return label,[make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def get_subspace(args,real_label, init_z, vis_flag=False):
    std = args.subspace_std
    bs = args.batch if not vis_flag else args.n_sample
    ind = np.random.randint(0, init_z.size(0), size=bs)
    z = init_z[ind]  # should give a tensor of size [batch_size, 512]
    for i in range(z.size(0)):
        for j in range(z.size(1)):
            z[i][j].data.normal_(z[i][j], std)
    ind = np.random.randint(0, real_label.size(0), size=bs)
    c = real_label[ind]
    for i in range(c.size(0)):
        for j in range(c.size(1)):
            c[i][j].data.normal_(c[i][j],std)
    return c,[z]

def show_samples(n,device):
    x=np.linspace(0.,1.,n+1,endpoint=False)[1:n+1]
    y=np.linspace(0.,1.,n+1,endpoint=False)[1:n+1]
    samples=np.array(np.meshgrid(x,y)).T.reshape(-1,2)
    return torch.tensor(samples,device=device).float()


def train(args, loader, generator, discriminator, extra, g_optim, d_optim, e_optim, g_ema, device, g_source, d_source,extra_patch,d_optim_patch,e_optim_patch):
    loader = sample_data(loader)

    imsave_path = os.path.join('samples', args.exp)
    model_path = os.path.join('checkpoints', args.exp)

    if not os.path.exists(imsave_path):
        os.makedirs(imsave_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # this defines the anchor points, and when sampling noise close to these, we impose image-level adversarial loss (Eq. 4 in the paper)
    init_z = torch.randn(args.n_train, args.latent, device=device)
    pbar = range(args.iter)
    sfm = nn.Softmax(dim=1)
    kl_loss = nn.KLDivLoss()
    sim = nn.CosineSimilarity()
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter,
                    dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}


    g_module = generator
    d_module = discriminator
    g_ema_module = g_ema.module

    accum = 0.5 ** (32 / (10 * 1000))
    ada_augment = torch.tensor([0.0, 0.0], device=device)
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    ada_aug_step = args.ada_target / args.ada_length
    r_t_stat = 0

    # this defines which level feature of the discriminator is used to implement the patch-level adversarial loss: could be anything between [0, args.highp] 
    lowp, highp = 0, args.highp

    # the following defines the constant noise used for generating images at different stages of training
    sample_z = torch.randn(4, args.latent, device=device)
    sample_c = show_samples(2,device)

    requires_grad(g_source, False)
    requires_grad(d_source, False)
    #sub_region_z = get_subspace(args, init_z.clone(), vis_flag=True)
    for idx in pbar:
        i = idx + args.start_iter
        which = i % args.subspace_freq # defines whether we sample from anchor region in this iteration or other

        if i > args.iter:
            print("Done!")
            break

        # load the training image and label
        real_img, real_label = next(loader)
        real_img = real_img.to(device)
        #print(real_img.shape)
        real_label = real_label.to(device)
        '''save_image(
                        real_img[0:4],
                        real_label,
                        os.path.join("samples/",args.exp,f"{str(i).zfill(6)}_true"),
                        nrow=1,
                        normalize=False,
                        range=(-1, 1),
                        use_wandb=False,
                        iteration=i
                    )'''
        

        # from here on, we optimise full discriminator
        requires_grad(generator, False)
        requires_grad(d_source, False)
        requires_grad(discriminator, True)
        requires_grad(extra, True)
        requires_grad(extra_patch, False)
        requires_named_grad(discriminator,['convs.0','convs.1'],False)

        if which > 0:
            # sample normally, apply patch-level adversarial loss
            fake_label,noise = mixing_noise(args.batch,args.paramdim, args.latent, args.mixing, device)
        else:
            # sample from anchors, apply image-level adversarial loss
            fake_label,noise = get_subspace(args, real_label.clone(),init_z.clone())
        #which = 0
        fake_img, _ = generator(fake_label,noise)

        if args.augment :
            real_img, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred, _ = discriminator(
            fake_img,fake_label, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp))
        real_pred, _ = discriminator(
            real_img,real_label, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp), real=True)
        #print(fake_pred.shape)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        extra.zero_grad()
        d_loss.backward()
        d_optim.step()
        e_optim.step()

        if args.augment and args.augment_p == 0:
            ada_augment += torch.tensor(
                (torch.sign(real_pred).sum().item(), real_pred.shape[0]), device=device
            )
            ada_augment = reduce_sum(ada_augment)

            if ada_augment[1] > 255:
                pred_signs, n_pred = ada_augment.tolist()

                r_t_stat = pred_signs / n_pred

                if r_t_stat > args.ada_target:
                    sign = 1

                else:
                    sign = -1

                ada_aug_p += sign * ada_aug_step * n_pred
                ada_aug_p = min(1, max(0, ada_aug_p))
                ada_augment.mul_(0)

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred, _ = discriminator(
                real_img,real_label, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp))
            real_pred = real_pred.view(real_img.size(0), -1)
            real_pred = real_pred.mean(dim=1).unsqueeze(1)

            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            extra.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every +
             0 * real_pred[0]).backward()

            d_optim.step()
            e_optim.step()
        loss_dict["r1"] = r1_loss


        #from here on, we update the source size discriminator
        requires_grad(generator, False)
        requires_grad(d_source, True)
        requires_grad(discriminator, False)
        requires_named_grad(d_source,['convs.0','convs.1'],False)
        requires_grad(extra, False)
        requires_grad(extra_patch, True)
        if args.patch and i%4==0:
            if which > 0:
                # sample normally, apply patch-level adversarial loss
                fake_label,noise = mixing_noise(args.batch,args.paramdim, args.latent, args.mixing, device)
            else:
                # sample from anchors, apply image-level adversarial loss
                fake_label,noise = get_subspace(args, real_label.clone(),init_z.clone())
            
            fake_label,noise = mixing_noise(args.batch,args.paramdim, args.latent, args.mixing, device)
            fake_img, _ = generator(fake_label,noise)

            if args.augment :
                real_img, _ = augment(real_img, ada_aug_p)
                fake_img, _ = augment(fake_img, ada_aug_p)

            patch_ind = random.randint(0,3*args.size[0]-1)
            patch_ind1 = random.randint(0,3*args.size[0]-1)

            fake_img = fake_img[:,:,patch_ind:patch_ind+args.size[0],:]
            real_img_patch = real_img[:,:,patch_ind1:patch_ind1+args.size[0],:]

            fake_pred, _ = d_source(
                fake_img,fake_label, extra=extra_patch, flag=which, p_ind=np.random.randint(lowp, highp))
            real_pred, _ = d_source(
                real_img_patch,real_label, extra=extra_patch, flag=which, p_ind=np.random.randint(lowp, highp), real=True)
            #print(fake_pred.shape)
            d_loss = d_logistic_loss(real_pred, fake_pred)
            #print(d_loss)

            loss_dict["d_patch"] = d_loss
            loss_dict["real_score_patch"] = real_pred.mean()
            loss_dict["fake_score_patch"] = fake_pred.mean()

            d_source.zero_grad()
            extra_patch.zero_grad()
            d_loss.backward()
            d_optim_patch.step()
            e_optim_patch.step()

            if args.augment and args.augment_p == 0:
                ada_augment += torch.tensor(
                    (torch.sign(real_pred).sum().item(), real_pred.shape[0]), device=device
                )
                ada_augment = reduce_sum(ada_augment)

                if ada_augment[1] > 255:
                    pred_signs, n_pred = ada_augment.tolist()

                    r_t_stat = pred_signs / n_pred

                    if r_t_stat > args.ada_target:
                        sign = 1

                    else:
                        sign = -1

                    ada_aug_p += sign * ada_aug_step * n_pred
                    ada_aug_p = min(1, max(0, ada_aug_p))
                    ada_augment.mul_(0)

            d_regularize = i % (args.d_reg_every*4) == 0

            if d_regularize:
                real_img_patch = real_img_patch.detach()
                real_img_patch.requires_grad = True
                real_pred, _ = d_source(
                    real_img_patch,real_label, extra=extra_patch, flag=which, p_ind=np.random.randint(lowp, highp))
                real_pred = real_pred.view(real_img.size(0), -1)
                real_pred = real_pred.mean(dim=1).unsqueeze(1)

                r1_loss = d_r1_loss(real_pred, real_img_patch)

                d_source.zero_grad()
                extra_patch.zero_grad()
                (args.r1 / 2 * r1_loss * args.d_reg_every +
                0 * real_pred[0]).backward()

                d_optim_patch.step()
                e_optim_patch.step()
            loss_dict["r1_patch"] = r1_loss

        #now update generator
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        requires_grad(extra, False)
        requires_grad(d_source, False)
        requires_grad(extra_patch, False)
        #requires_named_grad(generator,['style','preprocess'],False)
        if which > 0:
            # sample normally, apply patch-level adversarial loss
            fake_label,noise = mixing_noise(args.batch,args.paramdim, args.latent, args.mixing, device)
        else:
            # sample from anchors, apply image-level adversarial loss
            if args.full_anchor_p > random.random():
                fake_label,noise = get_subspace(args, real_label.clone(),init_z.clone())
            else :
                fake_label,_ = get_subspace(args, real_label.clone(),init_z.clone())
                _,noise = mixing_noise(args.batch,args.paramdim, args.latent, args.mixing, device)

        fake_img, _ = generator(fake_label,noise)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred, _ = discriminator(
            fake_img,fake_label, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp))
        g_loss = g_nonsaturating_loss(fake_pred)
        if args.patch:
            patch_ind = random.randint(0,3*args.size[0]-1)
            fake_img_patch = fake_img[:,:,patch_ind:patch_ind+args.size[0],:]
            fake_pred_patch,_ = d_source(
                fake_img_patch,fake_label, extra=extra_patch, flag=0, p_ind=np.random.randint(lowp, highp))
            g_loss_patch = g_nonsaturating_loss(fake_pred_patch)
        # distance consistency loss
        with torch.set_grad_enabled(False):
            z = torch.randn(args.feat_const_batch, args.latent, device=device)
            cond = torch.rand(args.feat_const_batch, args.paramdim, device=device)
            feat_ind = numpy.random.randint(1, g_source.module.n_latent - 1, size=args.feat_const_batch)

            # computing source distances
            source_sample, feat_source = g_source(cond,[z], return_feats=True)
            dist_source = torch.zeros(
                [args.feat_const_batch, args.feat_const_batch - 1]).cuda()

            # iterating over different elements in the batch
            for pair1 in range(args.feat_const_batch):
                tmpc = 0
                # comparing the possible pairs
                for pair2 in range(args.feat_const_batch):
                    if pair1 != pair2:
                        anchor_feat = torch.unsqueeze(
                            feat_source[feat_ind[pair1]][pair1].reshape(-1), 0)
                        compare_feat = torch.unsqueeze(
                            feat_source[feat_ind[pair1]][pair2].reshape(-1), 0)
                        dist_source[pair1, tmpc] = sim(
                            anchor_feat, compare_feat)
                        tmpc += 1
            dist_source = sfm(dist_source)

        # computing distances among target generations
        _, feat_target = generator(cond,[z], return_feats=True)
        dist_target = torch.zeros(
            [args.feat_const_batch, args.feat_const_batch - 1]).cuda()

        # iterating over different elements in the batch
        for pair1 in range(args.feat_const_batch):
            tmpc = 0
            for pair2 in range(args.feat_const_batch):  # comparing the possible pairs
                if pair1 != pair2:
                    anchor_feat = torch.unsqueeze(
                        feat_target[feat_ind[pair1]][pair1].reshape(-1), 0)
                    compare_feat = torch.unsqueeze(
                        feat_target[feat_ind[pair1]][pair2].reshape(-1), 0)
                    dist_target[pair1, tmpc] = sim(anchor_feat, compare_feat)
                    tmpc += 1
        dist_target = sfm(dist_target)
        rel_loss = args.kl_wt * \
            kl_loss(torch.log(dist_target), dist_source) # distance consistency loss 
        #print(g_loss)
        #print(rel_loss)
         
        if args.patch:
            g_loss = (g_loss + g_loss_patch)/2.
            #print(g_loss_patch)
        g_loss = g_loss + rel_loss

        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        # to save up space
        del rel_loss, g_loss, d_loss, fake_img,real_img, fake_pred, real_pred, anchor_feat, compare_feat, dist_source, dist_target, feat_source, feat_target

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            fake_label,noise = mixing_noise(
                path_batch_size,args.paramdim, args.latent, args.mixing, device)
            fake_img, latents = generator(fake_label,noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema_module, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        if args.patch:
            r1_val_patch = loss_reduced["r1_patch"].mean().item()
            d_patch_loss_val = loss_reduced["d_patch"].mean().item()
            real_score_val_patch = loss_reduced["real_score_patch"].mean().item()
            fake_score_val_patch = loss_reduced["fake_score_patch"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )
                if args.patch:
                    wandb.log(
                    {
                        "R1_patch": r1_val_patch,
                        "Patch Real Score": real_score_val_patch,
                        "Patch Fake Score": fake_score_val_patch,
                    }
                )

            if i % args.img_freq == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema(sample_c,[sample_z])
                    save_image(
                        sample[0:4],
                        sample_c,
                        os.path.join("samples/",args.exp,f"{str(i).zfill(6)}"),
                        nrow=1,
                        normalize=False,
                        range=(-1, 1),
                        use_wandb=False,
                        iteration=i
                    )
                    
                if i % (args.save_freq) == 0 and wandb:
                    with torch.no_grad():
                        #g_ema.eval()
                        #sample, _ = g_ema(sample_c,[sample_z])
                        save_image(
                            sample[0:4],
                            sample_c,
                            os.path.join("samples/",args.exp,f"{str(i).zfill(6)}"),
                            nrow=1,
                            normalize=False,
                            range=(-1, 1),
                            use_wandb=args.wandb,
                            iteration=i
                        )
                del sample

            if (i % args.save_freq == 0) and (i > 0):
                torch.save(
                    {
                        "g_ema": g_ema.state_dict(),
                        # uncomment the following lines only if you wish to resume training after saving. Otherwise, saving just the generator is sufficient for evaluations

                        #"g": g_module.state_dict(),
                        #"g_s": g_source.state_dict(),
                        #"d": d_module.state_dict(),
                        #"g_optim": g_optim.state_dict(),
                        #"d_optim": d_optim.state_dict(),
                    },
                    f"%s/{str(i).zfill(6)}.pt" % (model_path),
                )


if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--iter", type=int, default=5002)
    parser.add_argument("--save_freq", type=int, default=200)
    parser.add_argument("--img_freq", type=int, default=200)
    parser.add_argument("--kl_wt", type=int, default=1000)
    parser.add_argument("--highp", type=int, default=2)
    parser.add_argument("--subspace_freq", type=int, default=2)
    parser.add_argument("--feat_ind", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--feat_const_batch", type=int, default=8)
    parser.add_argument("--n_sample", type=int, default=25)
    parser.add_argument("--size", type=str, default=256)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--feat_res", type=int, default=128)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--path_regularize", type=float, default=2)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=8)
    parser.add_argument("--g_reg_every", type=int, default=2)
    parser.add_argument("--mixing", type=float, default=0.9)
    parser.add_argument("--subspace_std", type=float, default=0.05)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--source_key", type=str, default='ffhq')
    parser.add_argument("--exp", type=str, default=None, required=True)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--augment", dest='augment', action='store_true')
    parser.add_argument("--no-augment", dest='augment', action='store_false')
    parser.add_argument("--augment_p", type=float, default=0.0)
    parser.add_argument("--ada_target", type=float, default=0.6)
    parser.add_argument("--ada_length", type=int, default=500 * 1000)
    parser.add_argument("--n_train", type=int, default=10)
    parser.add_argument("--full_anchor_p", type=float, default=0.5)

    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    #n_gpu = os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else 1
    n_gpu = len([s.strip() for s in os.environ["CUDA_VISIBLE_DEVICES"].split(",")] ) if "CUDA_VISIBLE_DEVICES" in os.environ else 1
    #print(n_gpu)
    sizes = [int(s.strip()) for s in args.size.split(",")]
    args.size=sizes
    args.distributed = n_gpu > 1

    args.paramdim = 2
    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    generator = Generator(
        args.size[0], args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier,extend=True
    ).to(device)
    g_source = Generator(
        args.size[0], args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size[0], channel_multiplier=args.channel_multiplier
    ).to(device)
    d_source = Patch_Patch_Discriminator(
        args.size[0], channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = Generator(
        args.size[0], args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier,extend=True
    ).to(device)
    extra = Extra().to(device)
    extra_patch = Extra().to(device)
    #print(generator)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    d_optim_patch = optim.Adam(
        d_source.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    e_optim = optim.Adam(
        extra.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    e_optim_patch = optim.Adam(
        extra_patch.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )


    module_source = ['landscapes', 'red_noise',
                     'white_noise', 'hands', 'mountains', 'handsv2']
    
    if args.ckpt is not None:
        print("load model:", args.ckpt)
        assert args.source_key in args.ckpt
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        ckpt_source = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass


        generator.load_state_dict(ckpt["g"], strict=False)
        g_source.load_state_dict(ckpt_source["g"], strict=False)
        g_ema.load_state_dict(ckpt["g_ema"], strict=False)

        #d_source = nn.parallel.DataParallel(d_source)
        #discriminator = nn.parallel.DataParallel(discriminator)
        discriminator.load_state_dict(ckpt["d"], strict=False)
        d_source.load_state_dict(ckpt_source["d"], strict=False)

        #if 'g_optim' in ckpt.keys():
        #    g_optim.load_state_dict(ckpt["g_optim"], strict=False)
        #if 'd_optim' in ckpt.keys():
        #    d_optim.load_state_dict(ckpt["d_optim"], strict=False)
        #    d_optim_patch.load_state_dict(ckpt["d_optim"], strict=False)

    init_generator(generator)
    init_generator(g_ema)
    init_discriminator(discriminator)

    if args.distributed:
        geneator = nn.parallel.DataParallel(generator)
        g_ema = nn.parallel.DataParallel(g_ema)
        g_source = nn.parallel.DataParallel(g_source)

        discriminator = nn.parallel.DataParallel(discriminator)
        d_source = nn.parallel.DataParallel(d_source)
        extra = nn.parallel.DataParallel(extra)
        extra_patch = nn.parallel.DataParallel(extra_patch)

    transform = transforms.Compose(
        [
            #RandomFlip(),
            #RandomRotate(),
            transforms.ToTensor(),
            #transforms.Normalize(
            #    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    if len(sizes)>2:
        print('Image size should be 1 or 2 numbers')
        exit

    dataset = MultiResolutionDataset(args.data_path, transform, [args.size[0]*4,args.size[1]])
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=False),
        drop_last=True,
        num_workers=4,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="few shot lc expand", entity="dkn16")


    train(args, loader, generator, discriminator, extra, g_optim,
          d_optim, e_optim, g_ema, device, g_source, d_source,extra_patch,d_optim_patch,e_optim_patch)
