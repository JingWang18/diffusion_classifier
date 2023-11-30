import os
import argparse
import tqdm
from itertools import chain
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

import numpy as np

from utils import VisDAImage, weights_init, print_args
from model import ResBase101, ResBase50, ResClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default='/home/wang0918/projects/def-cdesilva/wang0918/overfit_domain_adapt/data/visda-2017')
parser.add_argument("--source", default="train")
parser.add_argument("--target", default="validation")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--shuffle", default=True)
parser.add_argument("--num_workers", default=0)
parser.add_argument("--epoch", default=30, type=int)
parser.add_argument("--snapshot", default="")
parser.add_argument("--lr", default=1e-3)
parser.add_argument("--class_num", default=12)
parser.add_argument("--extract", default=True)
parser.add_argument("--weight_L2norm", default=0.01)
parser.add_argument("--model", default='resnet50', type=str)
parser.add_argument("--post", default='-1', type=str)
parser.add_argument("--repeat", default='-1', type=str)
parser.add_argument("--result", default='/home/wang0918/projects/def-cdesilva/wang0918/diffusion_classifier/results')
args = parser.parse_args()

print_args(args)

source_root = os.path.join(args.data_root, args.source)
source_label = os.path.join(args.data_root, args.source + "_list.txt")
target_root = os.path.join(args.data_root, args.target)
target_label = os.path.join(args.data_root, args.target + "_list.txt")
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
source_set = VisDAImage(source_root, source_label, train_transform)
target_set = VisDAImage(target_root, target_label, train_transform)
assert len(source_set) == 152397
assert len(target_set) == 55388
source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers, drop_last=True)
target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size,
    shuffle=args.shuffle, num_workers=args.num_workers, drop_last=True)


result = open(os.path.join(args.result, "VisDA_IAFN_" + args.target + '_' + args.post + '.' + args.repeat + "_score.txt"), "a")

if args.model == 'resnet101':
    netG = ResBase101().cuda()
elif args.model == 'resnet50':
    netG = ResBase50().cuda()
else:
    raise ValueError('Unexpected value of args.model')
    
netF = ResClassifier(class_num=args.class_num, extract=args.extract).cuda()
netF.apply(weights_init)


def get_cls_loss(pred, gt):
    cls_loss = F.nll_loss(F.log_softmax(pred), gt)
    return cls_loss

def get_L2norm_loss_self_driven(x):
    radius = x.norm(p=2, dim=1).detach()
    assert radius.requires_grad == False
    radius = radius + 0.3
    l = ((x.norm(p=2, dim=1) - radius) ** 2).mean()
    return args.weight_L2norm * l

opt_g = optim.SGD(netG.parameters(), lr=args.lr, weight_decay=0.0005)
opt_f = optim.SGD(netF.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)

feat_bank = torch.randn(len(target_set), 2048) # can ca

#initialize
print("Initialize...")
with torch.no_grad():
    iter_target = iter(target_loader)
	# for i in range(len(target_loader)):
    for i in range(1):
        data = next(iter_target)
        inputs = data[0]
        indx = data[-1]
        inputs = inputs.cuda()
        feature = netG(inputs)
        feat_bank[indx] = feature.detach().clone().cpu()

print("Initialization Done!")
print(feat_bank.shape)

for epoch in range(1, args.epoch+1):
    source_loader_iter = iter(source_loader)
    target_loader_iter = iter(target_loader)
    print('>>training epoch : ' + str(epoch))

    step = epoch/args.epoch
    
    for i, (t_imgs, _, t_index) in tqdm.tqdm(enumerate(target_loader_iter)):
        s_imgs, s_labels, s_index = next(source_loader_iter)
        s_imgs = Variable(s_imgs.cuda())
        s_labels = Variable(s_labels.cuda())
        t_imgs = Variable(t_imgs.cuda())
        
        opt_g.zero_grad()
        opt_f.zero_grad()

        loss = 0

        s_bottleneck = netG(s_imgs)
        t_bottleneck = netG(t_imgs)

        with torch.no_grad():
            feat_bank[t_index] = t_bottleneck.detach().clone().cpu()


        similarity_s = F.normalize(s_bottleneck).cpu() @ F.normalize(feat_bank).T # batch*num_sample
        _, idx_source = torch.topk(similarity_s,
                                     dim=-1,
                                     largest=True,
                                     k= 6)
        idx_source = idx_source[:,1:]
        mu_s = torch.mean(feat_bank[idx_source], 1)
        std_s = torch.std(feat_bank[idx_source], 1)

        similarity_t = F.normalize(t_bottleneck).cpu() @ F.normalize(feat_bank).T # batch*num_sample
        _, idx_target = torch.topk(similarity_t,
                                     dim=-1,
                                     largest=True,
                                     k= 6)
        idx_target = idx_target[:,1:]
        mu_t = torch.mean(feat_bank[idx_target], 1)
        std_t = torch.std(feat_bank[idx_target], 1)


        s_fc2_emb, s_logit = netF(x=s_bottleneck, step=step, mu=mu_s, std=std_s, reverse=False)
        t_fc2_emb, t_logit = netF(x=t_bottleneck, step=step, mu=mu_t, std=std_t, reverse=True)

        ############################################################
        #################### Contribution ##########################
        ############################################################
        # Retain gradients for non-leaf tensors
        # d1_s.retain_grad()
        # d0_s.retain_grad()

        # # Choose a scalar function for the output for backpropagation (e.g., mean of y)
        # d1_scalar = d1_s.mean()  

        # # Compute the gradient of the output with respect to y and latent
        # d1_scalar.backward(retain_graph=True)

        # # Check the gradients
        # latent_gradient = d0_s.grad
        # print("Gradient with respect to latent:", latent_gradient.shape)

        # loss += 0.1*nn.MSELoss()(score_s, latent_gradient)

        ############################################################
        #################### Contribution ##########################
        ############################################################

        s_cls_loss = get_cls_loss(s_logit, s_labels)
        s_fc2_L2norm_loss = get_L2norm_loss_self_driven(s_fc2_emb)
        t_fc2_L2norm_loss = get_L2norm_loss_self_driven(t_fc2_emb)

        loss += s_cls_loss + s_fc2_L2norm_loss + t_fc2_L2norm_loss
        loss.backward()

        opt_g.step()
        opt_f.step()

    ##########################
    # Evaluation every epoch #
    ##########################
    netG.eval()
    netF.eval()

    correct = 0
    tick = 0
    subclasses_correct = np.zeros(args.class_num)
    subclasses_tick = np.zeros(args.class_num)
    
    for (imgs, labels, _) in target_loader:
        tick += 1
        imgs = Variable(imgs.cuda())
        feat = netG(imgs)
        similarity_t = F.normalize(feat).cpu() @ F.normalize(feat_bank).T # batch*num_sample
        _, idx_near = torch.topk(similarity_t,
                                     dim=-1,
                                     largest=True,
                                     k= 6)
        idx_near = idx_near[:,1:]
        mu_t = torch.mean(feat_bank[idx_near], 1)
        std_t = torch.std(feat_bank[idx_near], 1)

        _, pred = netF(x=feat, step=1.0, mu=mu_t, std=std_t, reverse=True)
        pred = F.softmax(pred)
        pred = pred.data.cpu().numpy()
        pred = pred.argmax(axis=1)
        labels = labels.numpy()
        for i in range(pred.size):
            subclasses_tick[labels[i]] += 1
            if pred[i] == labels[i]:
                correct += 1
                subclasses_correct[pred[i]] += 1

    correct = correct * 1.0 / len(target_set)
    subclasses_result = np.divide(subclasses_correct, subclasses_tick)

    print("Epoch {0}: {1}".format(epoch, correct))
    result.write("Epoch " + str(epoch) + ": " + str(correct) + "\n")
    # pre class accuracy
    for i in range(args.class_num):
        print("\tClass {0} : {1}".format(i, subclasses_result[i]))
        result.write("\tClass " + str(i) + ": " + str(subclasses_result[i]) + "\n")

    result.write("\tAvg : " + str(subclasses_result.mean()) + "\n")
    print("\tAvg : {0}\n".format(subclasses_result.mean()))

    torch.save(netG.state_dict(), os.path.join(args.snapshot, "VisDA_IAFN_"+ args.model + "_netG_" + args.post + '.' + args.repeat + '_' + str(epoch) + ".pth"))
    torch.save(netF.state_dict(), os.path.join(args.snapshot, "VisDA_IAFN_"+ args.model + "_netF_" + args.post + '.' + args.repeat + '_' + str(epoch) + ".pth"))

    netG.train()
    netF.train()

result.close()