import argparse
import os
import random
import shutil
from os.path import join
from torch.cuda.amp import autocast
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets.folder import ImageFolder
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import torch
from LabelSmoothing import LabelSmoothingLoss
import wandb
import gc

wandb.init()

from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', default='/media/cvpr/CM_1/datasets/original_dataset/cars_dMix',
                    help='dataset dir')
parser.add_argument('--mixing-set', default='/media/cvpr/CM_24/IPMix/IPMix-set/', help='Path to mixing set',
                    required=False)
parser.add_argument(
    '--no-jsd',
    '-nj',
    action='store_true',
    help='Turn off JSD consistency loss.')
parser.add_argument(
    '--beta',
    default=4,
    type=int,
    help='Severity of mixing')
parser.add_argument('-b', '--batch_size', default=64, help='batch_size')
parser.add_argument(
    '-g', '--gpu', default='0', help='example: 0 or 1, to use different gpu'
)
parser.add_argument('--alpha', default=1.0, type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('-w', '--num_workers', default=12, help='num_workers of dataloader')
parser.add_argument('-s', '--seed', default=2020, help='random seed')
parser.add_argument(
    '-n',
    '--note',
    default='',
    help='exp note, append after exp folder, fgvc(_r50) for example',
)
parser.add_argument(
    '-a',
    '--amp',
    default=2,
    help='0: w/o amp, 1: w/ nvidia apex.amp, 2: w/ torch.cuda.amp',
)
args = parser.parse_args()


def save_mixup_images(images, epoch, batch_idx, directory="saved_images"):
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Convert the tensor to grid image and save
    grid = torchvision.utils.make_grid(images)
    save_image(grid, f"{directory}/epoch_{epoch}_batch_{batch_idx}.png")


##### exp setting
seed = int(args.seed)
datasets_dir = args.dir
nb_epoch = 128  # 128 as default to suit scheduler
batch_size = int(args.batch_size)
num_workers = int(args.num_workers)
lr_begin = (batch_size / 256) * 0.1  # learning rate at begining
use_amp = int(args.amp)  # use amp to accelerate training

# wandb
config = wandb.config
config.batch_size = int(args.batch_size)
config.epochs = nb_epoch
config.learning_rate = lr_begin

##### data settings
data_dir = join('data', datasets_dir)
data_sets = ['train', 'val']
nb_class = len(
    os.listdir(join(data_dir, data_sets[0]))
)  # get number of class via img folders automatically
print("N. of Classes:", nb_class)
exp_dir = 'result/{}{}'.format(datasets_dir, args.note)  # the folder to save model

##### CUDA device setting
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

##### Random seed setting
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

re_size = 512
crop_size = 448

train_transform = transforms.Compose(
    [
        transforms.Resize((re_size, re_size)),
        transforms.RandomCrop(crop_size, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize((re_size, re_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # if i uncomment normalize the accuracy degrades from 86.15 to 85.47 on Res50 Car dataset
    ]
)

train = ImageFolder(root=join(data_dir, data_sets[0]), transform=train_transform)
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

transform = transforms.Compose([
    transforms.Resize((448, 448)),  # Reduced size
    transforms.ToTensor(),
])

fractal_img_dir = "/media/cvpr/CM_1/pytorch-image-classification/fractal/supp/fractal/0"
fractal_img_paths = [os.path.join(fractal_img_dir, fname) for fname in os.listdir(fractal_img_dir) if
                     fname.endswith(('.png', '.jpg', '.jpeg'))]
fractal_imgs = torch.stack([transform(Image.open(path).convert('RGB')) for path in fractal_img_paths])

##### Model settings
net = torchvision.models.resnet18(pretrained=False)
net.fc = nn.Linear(net.fc.in_features, nb_class)
for param in net.parameters():
    param.requires_grad = True  # make parameters in model learnable

##### optimizer setting
# criterion = LabelSmoothingLoss(classes=nb_class, smoothing=0.1)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr_begin, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=128)

##### file/folder prepare
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

shutil.copyfile('train.py', exp_dir + '/train.py')
shutil.copyfile('LabelSmoothing.py', exp_dir + '/LabelSmoothing.py')

with open(os.path.join(exp_dir, 'train_log.csv'), 'w+') as file:
    file.write('Epoch, lr, Train_Loss, Train_Acc, Test_Acc\n')

##### Apex
if use_amp == 1:  # use nvidia apex.amp
    print('\n===== Using NVIDIA AMP =====')
    from apex import amp

    net.cuda()
    net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
    with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
        file.write('===== Using NVIDIA AMP =====\n')
elif use_amp == 2:  # use torch.cuda.amp
    print('\n===== Using Torch AMP =====')
    from torch.cuda.amp import GradScaler, autocast

    scaler = GradScaler()
    with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
        file.write('===== Using Torch AMP =====\n')


# Replace the mixup_data function call in your training loop with cutmix_data


def mixup_data(source_images, source_labels, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = source_images.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    target_images = source_images[index, :]
    blended_images = lam * source_images + (1 - lam) * target_images
    blended_labels_a, blended_labels_b = source_labels, source_labels[index]
    return blended_images, blended_labels_a, blended_labels_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


##### 2 - Training #####
########################
net.cuda()
min_train_loss = float('inf')
max_eval_acc = 0
saving_interval = 2

for epoch in range(nb_epoch):
    print('\n===== Epoch: {} ====='.format(epoch))
    net.train()  # set model to train mode, enable Batch Normalization and Dropout
    lr_now = optimizer.param_groups[0]['lr']
    train_loss = train_correct = train_total = idx = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, ncols=80)):
        idx = batch_idx

        if inputs.shape[0] < batch_size:
            continue

        optimizer.zero_grad()  # Sets the gradients to zero
        inputs, targets = inputs.cuda(), targets.cuda()

        ##### amp setting
        if use_amp == 1:  # use nvidia apex.amp
            x = net(inputs)
            loss = LSLoss(x, targets)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        elif use_amp == 2:  # use torch.cuda.amp
            with autocast():
                clean_inputs, clean_targets = inputs.cuda(), targets.cuda()
                mixed_inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, use_cuda=True)
                mixed_inputs, targets_a, targets_b = map(Variable, (mixed_inputs, targets_a, targets_b))

                mixed_outputs = net(mixed_inputs)

                # save_directory = 'blended'
                #
                # # Save the images. Here's how to save the entire batch as a grid:
                # grid_image = torchvision.utils.make_grid(mixed_inputs.cpu(),
                #                                          nrow=4)  # Adjust nrow as per your batch size or preference
                # torchvision.utils.save_image(grid_image, f"{save_directory}/epoch_{epoch}_batch_{batch_idx}.png")
                #
                # # If you want to save each image separately, you can do something like this:
                # for i, image in enumerate(mixed_inputs):
                #     torchvision.utils.save_image(image.cpu(),
                #                                  f"{save_directory}/epoch_{epoch}_batch_{batch_idx}_image_{i}.png")

                clean_outputs = net(clean_inputs)
                softmax_clean_outputs = F.softmax(clean_outputs, dim=1)

                primary_loss = mixup_criterion(criterion, mixed_outputs, targets_a, targets_b, lam)
                softmax_mixed_outputs = F.softmax(mixed_outputs, dim=1)

                p_mixture = torch.clamp((softmax_mixed_outputs + softmax_clean_outputs) / 2., 1e-7, 1).log()
                kl_divergence = F.kl_div(p_mixture, softmax_clean_outputs, reduction='batchmean')
                lambda_kl = 12  # You need to choose an appropriate value for lambda_kl
                total_loss = primary_loss + lambda_kl * kl_divergence

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            x = net(inputs)
            loss = LSLoss(x, targets)
            loss.backward()
            optimizer.step()

        _, predicted = torch.max(clean_outputs.data, 1)
        train_total += targets.size(0)
        train_correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                          + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
        train_loss += total_loss.item()

    scheduler.step()

    train_acc = 100.0 * float(train_correct) / train_total
    train_loss = train_loss / (idx + 1)
    print(
        'Train | lr: {:.4f} | Loss: {:.4f} | Acc: {:.3f}% ({}/{})'.format(
            lr_now, train_loss, train_acc, train_correct, train_total
        )
    )
    wandb.log({'train_loss': train_loss, 'train_accuracy': train_acc, 'epoch': epoch})

    ##### Evaluating model with test data every epoch
    with torch.no_grad():
        net.eval()  # set model to eval mode, disable Batch Normalization and Dropout
        eval_set = ImageFolder(
            root=join(data_dir, data_sets[-1]), transform=test_transform
        )
        eval_loader = DataLoader(
            eval_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        eval_correct = eval_total = 0
        for _, (inputs, targets) in enumerate(tqdm(eval_loader, ncols=80)):
            inputs, targets = inputs.cuda(), targets.cuda()
            x = net(inputs)
            _, predicted = torch.max(x.data, 1)
            eval_total += targets.size(0)
            eval_correct += predicted.eq(targets.data).cpu().sum()
        eval_acc = 100.0 * float(eval_correct) / eval_total
        wandb.log({'validation_accuracy': eval_acc, 'epoch': epoch})
        print(
            '{} | Acc: {:.3f}% ({}/{})'.format(
                data_sets[-1], eval_acc, eval_correct, eval_total
            )
        )

        ##### Logging
        with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
            file.write(
                '{}, {:.4f}, {:.4f}, {:.3f}%, {:.3f}%\n'.format(
                    epoch, lr_now, train_loss, train_acc, eval_acc
                )
            )

        ##### save model with highest acc
        if eval_acc > max_eval_acc:
            max_eval_acc = eval_acc
            torch.save(
                net.state_dict(),
                os.path.join(exp_dir, 'max_acc.pth'),
                _use_new_zipfile_serialization=False,
            )

########################
##### 3 - Testing  #####
########################
print('\n\n===== TESTING =====')

with open(os.path.join(exp_dir, 'train_log.csv'), 'a') as file:
    file.write('===== TESTING =====\n')

##### load best model
net.load_state_dict(torch.load(join(exp_dir, 'max_acc.pth')))
net.eval()  # set model to eval mode, disable Batch Normalization and Dropout

for data_set in data_sets:
    testset = ImageFolder(
        root=os.path.join(data_dir, data_set), transform=test_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    test_loss = correct = total = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(tqdm(testloader, ncols=80)):
            inputs, targets = inputs.cuda(), targets.cuda()
            x = net(inputs)
            _, predicted = torch.max(x.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
    test_acc = 100.0 * float(correct) / total
    print('Dataset {}\tACC:{:.2f}\n'.format(data_set, test_acc))

    wandb.log({'final_test_accuracy_{}'.format(data_set): test_acc})

    ##### logging
    with open(os.path.join(exp_dir, 'train_log.csv'), 'a+') as file:
        file.write('Dataset {}\tACC:{:.2f}\n'.format(data_set, test_acc))

    with open(
            os.path.join(exp_dir, 'acc_{}_{:.2f}'.format(data_set, test_acc)), 'a+'
    ) as file:
        # save accuracy as file name
        pass

wandb.finish()
