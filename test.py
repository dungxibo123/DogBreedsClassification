from model import training_process

opt = {
    'data_folder': './data/images/CloneImages',
    'num_classes': 20,
    'batch_size': 32,
    'val_batch_size': 128,
    'checkpoint_pth': './checkpoint/',
    'resume_from_checkpoint': False,
    'model': 'resnet152',
    'pretrained_weights': 'IMAGENET1K_V2',
    'epoch': 20,
    'early_stopping': True,
    'patience': 10,
    'optimizer': 'adam',
    'learning_rate': 0.005,
    'weight_decay': 4e-3,
    'use_gpu': False,
    'checkpoint_save_freq': 5,
    'validation_split': 0.2,
    'dropout': 0.35,
    'num_hidden': 512,
    'continual_checkpoint_pth': '/content/checkpoint/checkpoint_e15.pt'

}
losses, val_losses, train_acc, val_acc = training_process(opt)
