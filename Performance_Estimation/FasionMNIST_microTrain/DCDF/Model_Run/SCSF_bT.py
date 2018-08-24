from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np

from _cnn_model import DCDF_Net,weights_init
from _mixed_train_sgd import micro_train,all_train,bias_train

import time
from _data_process import asMinutesUnit

# Training settings
# ==============================================================================
parser = argparse.ArgumentParser(description='PyTorch FasionMNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--total_epochs', type=int, default=105, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--k_allTrain_epochs', type=int, default=0, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--conv1_size', type=int, default=64, metavar='N',
                    help='kernels of first convolutional layer (default: 160)')
parser.add_argument('--conv2_size', type=int, default=128, metavar='N',
                    help='kernels of second convolutional layer (default: 320)')
parser.add_argument('--fc1_size', type=int, default=512, metavar='N',
                    help='units of first fully-connected layer (default: 1024)')
parser.add_argument('--get_state', action='store_true', default=False,
                    help='get all model state of all training epochs')
# ------------------------------------------------------------------------------
# setting learning rate as https://cs.nyu.edu/~wanli/dropc/dropc.pdf.
parser.add_argument('--LR_window1', type=int, default=15, metavar='W1',
                    help='Adam learning rate window 1 (default: 150)')
parser.add_argument('--LR_window2', type=int, default=15, metavar='W2',
                    help='Adam learning rate window 2 (default: 100)')
parser.add_argument('--LR_window3', type=int, default=15, metavar='W3',
                    help='Adam learning rate window 3 (default: 50)')

parser.add_argument('--SGD_lr', type=float, default=0.001, metavar='LR',
                    help='SGD learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--Adam_lr', type=float, default=0.001, metavar='LR',
                    help='Initial Adam learning rate (default: 0.01)')

# ------------------------------------------------------------------------------
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
# ------------------------------------------------------------------------------
parser.add_argument('--seed', type=int, default=2, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=60, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_train', type=int, default=1, metavar='N',
                    help='how many training times to use (default: 1)')
# ------------------------------------------------------------------------------
parser.add_argument('--gpu', type=int, default=0, metavar='C',
                    help='random seed (default: 0)')
# ------------------------------------------------------------------------------
parser.add_argument('--model_name', type=str, default="", metavar='S',
                    help='model name')  
# ==============================================================================

if __name__ == '__main__':


    args = parser.parse_args()

    torch.manual_seed(args.seed)

    args.total_epochs = args.LR_window1 + args.LR_window2 *2 + args.LR_window3*4

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device=torch.device("cuda", args.gpu if use_cuda else "cpu")

    aT_model = DCDF_Net(conv1_size=args.conv1_size,conv2_size=args.conv2_size,fc1_size=args.fc1_size)
    aT_model.apply(weights_init)

    # aT_model.share_memory().to(device) # gradients are allocated lazily, so they are not shared here 
    model_name="C"+str(args.conv1_size)+"C"+str(args.conv2_size)+"F"+str(args.fc1_size)+"F10"

    args.model_name=model_name
    dir_model_state="../Model_State/"+model_name
    if os.path.exists(dir_model_state) == False:
        os.mkdir(dir_model_state)
    
    dir_result="../Result_npz/"+model_name
    if os.path.exists(dir_result) == False:
        os.mkdir(dir_result)
        
    time_start=time.time()
    if args.k_allTrain_epochs == args.total_epochs:
        train_acc_array,test_acc_array=all_train(args,aT_model,device)
    elif 0<=args.k_allTrain_epochs < args.total_epochs:
        # train_acc_array,test_acc_array=micro_train(args,aT_model,device)
        train_acc_array,test_acc_array=bias_train(args,aT_model,device)
    else :
        print("Error ! please make sure k_allTrain is smaller than totals!")
        exit()


    log_dirs = "../Result_npz/"+model_name
    
    with open(log_dirs+"/Time_Log.txt", "a+") as f:
        print("%d\t%s" % (args.k_allTrain_epochs,asMinutesUnit(time.time() - time_start)) , file=f)
    # np.savez(dirs+"/acc"+str(int(microtrain_steps/display_step))+".npz", test_acc_array, train_acc_array)
    np.savez(log_dirs+"/Acc_bT_"+str(args.k_allTrain_epochs)+".npz", test_acc_array, train_acc_array)

    os.rename("../Result_npz/"+args.model_name+"/TestLog_"+str(args.k_allTrain_epochs)+".txt","../Result_npz/"+args.model_name+"/TestLog_bT_"+str(args.k_allTrain_epochs)+".txt")