 # model.py
import torch.nn as nn
import torch
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torchvision
import os
import sys
import torch.nn.functional as F
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from data import *


model_dict = {
    'ResNet18': torchvision.models.resnet18,
    'ResNet34': torchvision.models.resnet34,
    'ResNet50': torchvision.models.resnet50,
    'ResNet101': torchvision.models.resnet101,
    'ResNet152': torchvision.models.resnet152
}
def create_model(opt):
  model_ft = model_dict[opt['model']](progress=False, weights=opt['pretrained_weights'])
  for param in model_ft.parameters():
    param.requires_grad = False
  num_ftrs = model_ft.fc.in_features


  model_ft.fc = nn.Sequential(
      nn.Linear(num_ftrs, opt['num_hidden']),
      nn.ReLU(),
      nn.Dropout(opt['dropout']),
      nn.Linear(opt['num_hidden'], opt['num_classes']),
      nn.Softmax(dim=1)
  )
  return model_ft

def criterion():
    return F.cross_entropy
def evaluate(model, val_loader, opt):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0
    count = 0
    with torch.no_grad():
        for batch_id , test_data in enumerate(val_loader,0):
            count += 1
            data, label = test_data
            #label = F.one_hot(label, num_classes=opt.num_classes)
            if opt['use_gpu']:
                data = data.to("cuda")
                label = label.to("cuda")
            outputs = model(data)
            _, correct_labels = torch.max(label, 1)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == correct_labels).sum().item()
            running_loss += criterion()(
                outputs.float(), label.float()).item()
    #        print(f"--> Total {total}\n-->batch_id: {batch_id + 1}")
    acc = round(correct/total * 1.0 , 5)
    running_loss /= count
    return running_loss, acc
def train_one_iter(model, optim, train_load, val_loader, opt, epoch, lr_scheduler):
    losses = 0
    total_correct = 0
    total_sample = 0
    model.train()
    with tqdm(train_load,  unit="batch", position=0, leave=True) as tp:
        tp.set_description(f"Epoch {epoch}/{opt['epoch']}")
        for (batch_x, batch_y) in tp:

            optim.zero_grad()
            #batch_y = torch.nn.functional.one_hot(batch_y, num_classes = opt.num_classes)
            if opt['use_gpu']:
                batch_x = batch_x.to("cuda")
                batch_y = batch_y.to("cuda")
            outputs = model(batch_x)
            _, correct_labels = torch.max(batch_y, 1)
            _, predicted = torch.max(outputs.data, 1)
            total = batch_y.size(0)
            correct = (predicted == correct_labels).sum().item()
            total_correct += correct
            total_sample += total
            train_acc = round(total_correct / total_sample, 5)
            loss = criterion()(outputs.float(), batch_y.float())
            loss_item = loss.item()
            losses += loss_item
            loss.backward()
            optim.step()
            tp.set_postfix(loss=loss_item, train_acc=train_acc)
        val_loss, val_acc = evaluate(model, val_loader, opt)
        lr_scheduler.step()
        losses /= len(tp)
        print(f"Epoch {epoch}/{opt['epoch']} is finished with validation accuracy is {val_acc}")

    return model, optim, losses, total_correct / total_sample, val_loss, val_acc
def load_from_checkpoint(option):
    ckpt = torch.load(option['continual_checkpoint_pth'])
    opt = ckpt['opt']
    model = create_model(opt)
    model = model.to(ckpt['device'])
    model.load_state_dict(ckpt['model_state_dict'])
    optim = None
    if opt['optimizer'] == "sgd":
        optim = torch.optim.SGD(model.parameters())
    elif opt['optimizer'] == "adam":
        optim = torch.optim.Adam(model.parameters())
    elif opt['optimizer'] == "adamw":
        optim = torch.optim.AdamW(model.parameters())
    optim.load_state_dict(ckpt['optimizer_state_dict'])

    return model,\
           optim,\
           ckpt['epoch'],\
           ckpt['losses'],\
           ckpt['val_losses'],\
           ckpt['train_acc'],\
           ckpt['val_acc'],\
           opt

def training_process(opt):
    train_loader, val_loader = getTrainValLoader(opt)

    os.makedirs(opt['checkpoint_pth'], exist_ok=True)
    if opt['resume_from_checkpoint']:
      model, optim, init_epoch, losses,  val_losses, train_accuracies, val_accuracies, opt = load_from_checkpoint(opt)
      #best_model = model
      #best_val_acc = ckpt
      #
    else:

      model = create_model(opt)
      best_model = None
      best_val_acc = 0
      best_epoch = -1
      losses = []
      val_accuracies = []
      val_losses = []
      init_epoch = 1
      train_accuracies = []

      optim = None
      if opt['optimizer'] == "sgd":
          optim = torch.optim.SGD(model.parameters(), lr=opt['learning_rate'], weight_decay=opt['weight_decay'], nesterov=True, momentum=0.9)
      elif opt['optimizer'] == "adam":
          optim = torch.optim.Adam(model.parameters(), lr=opt['learning_rate'], weight_decay=opt['weight_decay'], eps=1e-7)
      elif opt['optimizer'] == "adamw":
          optim = torch.optim.AdamW(model.parameters(), lr=opt['learning_rate'], weight_decay=opt['weight_decay'], eps=1e-7)
    if opt['use_gpu'] and torch.cuda.is_available():
        model = model.to("cuda")



    now = datetime.datetime.now()
    #tags = {"author": "Dung Vo", "data": opt.data}
    #runName = f"Data: {opt.data}_Optimization: {opt.optimizer}_LR: {opt.learning_rate}_Batch Size: {opt.batch_size}_DropOut: {opt.dropout}_{datetime.datetime.strftime(now, format='%Y%m%d%H%M%S')}"


    lr_scheduler = MultiStepLR(optim, milestones=[5, 10, 15, 20, 25, 30], gamma=0.2, verbose=True)
    #reduceLROnPlateau = ReduceLROnPlateau(optim, patience=2, factor=0.1)
    patience_cnt = 0
    for i in range(1, opt['epoch']+1):


        model, optim, epoch_loss, train_acc, val_loss, val_acc = train_one_iter(model,
                                                                                optim, train_loader,
                                                                                val_loader, opt, epoch = i,
                                                                                lr_scheduler=lr_scheduler
                                                                               )
        losses.append(epoch_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        train_accuracies.append(train_acc)
        if i % opt['checkpoint_save_freq'] == 0 and opt['save_checkpoint']:
            torch.save({
                'opt': opt,
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'losses': losses,
                'val_losses': val_losses,
                'train_acc': train_accuracies,
                'val_acc': val_accuracies,
                'device': "cuda" if opt['use_gpu'] and torch.cuda.is_available() else "cpu"
            }, opt['checkpoint_pth']  + f"/checkpoint_e{i}.pt")
        if val_acc > best_val_acc:
            best_model = model
            best_val_acc = val_acc
            best_epoch = i
            patience_cnt = 0 # Reset the patience_cnt
        else: # Increase the patience for early_stopping if metric does not improve
            patience_cnt += 1
            if patience_cnt > opt['patience']:
                break



    torch.save({
        'opt': opt,
        'model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'epoch': best_epoch,
        'losses': losses[:best_epoch],
        'val_losses': val_losses[:best_epoch],
        'train_acc': train_accuracies[:best_epoch],
        'val_acc': val_accuracies[:best_epoch],
        'device': "cuda" if opt['use_gpu'] and torch.cuda.is_available() else "cpu"

    }, opt['checkpoint_pth'] + "/final_model.pt")
    return best_model, losses, val_losses, train_accuracies, val_accuracies
