from dataset import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
import hydra
import omegaconf

def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc

@hydra.main(version_base=None, config_path='config', config_name='cls')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["hydra.job.chdir"] = "True"
    logger = logging.getLogger(__name__)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = hydra.utils.to_absolute_path('modelnet40_normal_resampled/')

    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    args.num_class = 40
    args.input_dim = 6 if args.normal else 3
    # shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    a = args.model.name

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerCls')(
        args).cuda()
    criterion = torch.nn.CrossEntropyLoss()

    checkpoint = torch.load('best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader)
        logger.info('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    main()
