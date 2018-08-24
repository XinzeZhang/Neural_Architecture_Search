import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np

import time
from _data_process import timeSince


def all_train(args, model, device):
    model = model.to(device)
    # model.share_memory().to(device)
    torch.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    train_acc_array = []
    # train_acc=0.0
    test_acc_array = []
    # test_acc=0.0

    init_lr = args.SGD_lr
    # optimizer = optim.SGD(model.parameters(),lr=args.aT_Adam_lr,momentum=args.momentum)

    # window counting
    w1 = args.LR_window1
    w12a = args.LR_window1+args.LR_window2
    w12b = args.LR_window1+args.LR_window2*2
    w123a = w12b+args.LR_window3
    w123b = w12b+args.LR_window3*2
    w123c = w12b+args.LR_window3*3
    w123d = w12b+args.LR_window3*4

    if args.get_state:
        k_model_name = args.model_name+"_0"
        dir_model_state = "../Model_State/"+args.model_name+"/"+k_model_name+".pkl"
        torch.save(model.state_dict(), dir_model_state)
        print("Save the model state: "+k_model_name+".pkl")

    # Initialize timer
    time_tr_start = time.time()
    # lr window 1
    train_window(time_tr_start, 0, w1, args, model, device, train_loader,
                 test_loader, init_lr, train_acc_array, test_acc_array)
    # lr window 2
    train_window(time_tr_start, w1, w12a, args, model, device, train_loader,
                 test_loader, init_lr*0.5, train_acc_array, test_acc_array)
    train_window(time_tr_start, w12a, w12b, args, model, device, train_loader,
                 test_loader, init_lr*0.1, train_acc_array, test_acc_array)
    # lr window 3
    train_window(time_tr_start, w12b, w123a, args, model, device, train_loader,
                 test_loader, init_lr*0.05, train_acc_array, test_acc_array)
    train_window(time_tr_start, w123a, w123b, args, model, device, train_loader,
                 test_loader, init_lr*0.01, train_acc_array, test_acc_array)
    train_window(time_tr_start, w123b, w123c, args, model, device, train_loader,
                 test_loader, init_lr*0.005, train_acc_array, test_acc_array)
    train_window(time_tr_start, w123c, w123d, args, model, device, train_loader,
                 test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    # model_name = str(model.__class__.__name__)+"_C"+str(args.n_kernel)+"F10"
    # k_model_name = model_name+"_"+str(args.k_allTrain_epochs)
    # dir_model_state = "../Model_State/"+model_name+"/"+k_model_name+".pkl"
    # torch.save(model.state_dict(), dir_model_state)

    return train_acc_array, test_acc_array


def micro_train(args, model, device):
    '''
    Training for some steps, say K, for the network, then fix the convolutional layer and only train the last linear layer works well.
    '''
    model = model.to(device)
    model.share_memory().to(device)
    torch.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    train_acc_array = []
    # train_acc=0.0
    test_acc_array = []
    # test_acc=0.0

    init_lr = args.SGD_lr
    # optimizer = optim.SGD(model.parameters(),lr=args.aT_Adam_lr,momentum=args.momentum)

    w1 = args.LR_window1
    w12a = args.LR_window1+args.LR_window2
    w12b = args.LR_window1+args.LR_window2*2
    w123a = w12b+args.LR_window3
    w123b = w12b+args.LR_window3*2
    w123c = w12b+args.LR_window3*3
    w123d = w12b+args.LR_window3*4

    # Initialize timer
    time_tr_start = time.time()
    # lr windows 1
    if args.k_allTrain_epochs < w1:
        conv_fixed(time_tr_start, 0, w1, args, model, device, train_loader,
                   test_loader, init_lr, train_acc_array, test_acc_array)
        train_window(time_tr_start, w1, w12a, args, model, device, train_loader,
                     test_loader, init_lr*0.5, train_acc_array, test_acc_array)
        train_window(time_tr_start, w12a, w12b, args, model, device, train_loader,
                     test_loader, init_lr*0.1, train_acc_array, test_acc_array)
        train_window(time_tr_start, w12b, w123a, args, model, device, train_loader,
                     test_loader, init_lr*0.05, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123a, w123b, args, model, device, train_loader,
                     test_loader, init_lr*0.01, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123b, w123c, args, model, device, train_loader,
                     test_loader, init_lr*0.005, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123c, w123d, args, model, device, train_loader,
                     test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    # lr windows 2
    if w1 <= args.k_allTrain_epochs < w12a:
        # train_window(time_tr_start,0, w1, args, model, device, train_loader,
        #              test_loader, init_lr, train_acc_array, test_acc_array)
        conv_fixed(time_tr_start, w1, w12a, args, model, device, train_loader,
                   test_loader, init_lr*0.5, train_acc_array, test_acc_array)
        train_window(time_tr_start, w12a, w12b, args, model, device, train_loader,
                     test_loader, init_lr*0.1, train_acc_array, test_acc_array)
        train_window(time_tr_start, w12b, w123a, args, model, device, train_loader,
                     test_loader, init_lr*0.05, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123a, w123b, args, model, device, train_loader,
                     test_loader, init_lr*0.01, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123b, w123c, args, model, device, train_loader,
                     test_loader, init_lr*0.005, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123c, w123d, args, model, device, train_loader,
                     test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    if w12a <= args.k_allTrain_epochs < w12b:
        # train_window(time_tr_start,0, w1, args, model, device, train_loader,
        #              test_loader, init_lr, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w1, w12a, args, model, device, train_loader,
        #              test_loader, init_lr*0.5, train_acc_array, test_acc_array)
        conv_fixed(time_tr_start, w12a, w12b, args, model, device, train_loader,
                   test_loader, init_lr*0.1, train_acc_array, test_acc_array)
        train_window(time_tr_start, w12b, w123a, args, model, device, train_loader,
                     test_loader, init_lr*0.05, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123a, w123b, args, model, device, train_loader,
                     test_loader, init_lr*0.01, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123b, w123c, args, model, device, train_loader,
                     test_loader, init_lr*0.005, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123c, w123d, args, model, device, train_loader,
                     test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    # lr windows 3
    if w12b <= args.k_allTrain_epochs < w123a:
        # train_window(time_tr_start,0, w1, args, model, device, train_loader,
        #              test_loader, init_lr, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w1, w12a, args, model, device, train_loader,
        #              test_loader, init_lr*0.5, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w12a, w12b, args, model, device, train_loader,
        #              test_loader, init_lr*0.1, train_acc_array, test_acc_array)
        conv_fixed(time_tr_start, w12b, w123a, args, model, device, train_loader,
                   test_loader, init_lr*0.05, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123a, w123b, args, model, device, train_loader,
                     test_loader, init_lr*0.01, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123b, w123c, args, model, device, train_loader,
                     test_loader, init_lr*0.005, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123c, w123d, args, model, device, train_loader,
                     test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    if w123a <= args.k_allTrain_epochs < w123b:
        # train_window(time_tr_start,0, w1, args, model, device, train_loader,
        #              test_loader, init_lr, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w1, w12a, args, model, device, train_loader,
        #              test_loader, init_lr*0.5, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w12a, w12b, args, model, device, train_loader,
        #              test_loader, init_lr*0.1, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w12b, w123a, args, model, device, train_loader,
        #             test_loader, init_lr*0.05, train_acc_array, test_acc_array)
        conv_fixed(time_tr_start, w123a, w123b, args, model, device, train_loader,
                   test_loader, init_lr*0.01, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123b, w123c, args, model, device, train_loader,
                     test_loader, init_lr*0.005, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123c, w123d, args, model, device, train_loader,
                     test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    if w123b <= args.k_allTrain_epochs < w123c:
        # train_window(time_tr_start,0, w1, args, model, device, train_loader,
        #              test_loader, init_lr, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w1, w12a, args, model, device, train_loader,
        #              test_loader, init_lr*0.5, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w12a, w12b, args, model, device, train_loader,
        #              test_loader, init_lr*0.1, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w12b, w123a, args, model, device, train_loader,
        #             test_loader, init_lr*0.05, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w123a, w123b, args, model, device, train_loader,
        #              test_loader, init_lr*0.01, train_acc_array, test_acc_array)
        conv_fixed(time_tr_start, w123b, w123c, args, model, device, train_loader,
                   test_loader, init_lr*0.005, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123c, w123d, args, model, device, train_loader,
                     test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    if w123c <= args.k_allTrain_epochs < w123d:
        # train_window(time_tr_start,0, w1, args, model, device, train_loader,
        #              test_loader, init_lr, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w1, w12a, args, model, device, train_loader,
        #              test_loader, init_lr*0.5, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w12a, w12b, args, model, device, train_loader,
        #              test_loader, init_lr*0.1, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w12b, w123a, args, model, device, train_loader,
        #             test_loader, init_lr*0.05, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w123a, w123b, args, model, device, train_loader,
        #              test_loader, init_lr*0.01, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w123b, w123c, args, model, device, train_loader,
        #              test_loader, init_lr*0.005, train_acc_array, test_acc_array)
        conv_fixed(time_tr_start, w123c, w123d, args, model, device, train_loader,
                   test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    # dir_model_state = "../Model_State/"+str(model.__class__.__name__)+"_C"+str(args.n_kernel)+"F10"+"/"+str(
    #     model.__class__.__name__)+"_C"+str(args.n_kernel)+"F10"+"_"+str(args.k_allTrain_epochs)+".pkl"
    # torch.save(model.state_dict(), dir_model_state)

    return train_acc_array, test_acc_array


def bias_train(args, model, device):
    '''
    Training for some steps, say K, for the network, then fix the weight of layers and only train the bias.
    return train_acc_array, test_acc_array
    '''
    model = model.to(device)
    model.share_memory().to(device)
    torch.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    train_acc_array = []
    # train_acc=0.0
    test_acc_array = []
    # test_acc=0.0

    init_lr = args.SGD_lr
    # optimizer = optim.SGD(model.parameters(),lr=args.aT_Adam_lr,momentum=args.momentum)

    w1 = args.LR_window1
    w12a = args.LR_window1+args.LR_window2
    w12b = args.LR_window1+args.LR_window2*2
    w123a = w12b+args.LR_window3
    w123b = w12b+args.LR_window3*2
    w123c = w12b+args.LR_window3*3
    w123d = w12b+args.LR_window3*4
    
    # Initialize timer
    time_tr_start = time.time()
    # lr windows 1
    if args.k_allTrain_epochs < w1:
        weight_fixed(time_tr_start, 0, w1, args, model, device, train_loader,
                     test_loader, init_lr, train_acc_array, test_acc_array)
        train_window(time_tr_start, w1, w12a, args, model, device, train_loader,
                     test_loader, init_lr*0.5, train_acc_array, test_acc_array)
        train_window(time_tr_start, w12a, w12b, args, model, device, train_loader,
                     test_loader, init_lr*0.1, train_acc_array, test_acc_array)
        train_window(time_tr_start, w12b, w123a, args, model, device, train_loader,
                     test_loader, init_lr*0.05, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123a, w123b, args, model, device, train_loader,
                     test_loader, init_lr*0.01, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123b, w123c, args, model, device, train_loader,
                     test_loader, init_lr*0.005, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123c, w123d, args, model, device, train_loader,
                     test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    # lr windows 2
    if w1 <= args.k_allTrain_epochs < w12a:
        # train_window(time_tr_start,0, w1, args, model, device, train_loader,
        #              test_loader, init_lr, train_acc_array, test_acc_array)
        weight_fixed(time_tr_start, w1, w12a, args, model, device, train_loader,
                     test_loader, init_lr*0.5, train_acc_array, test_acc_array)
        train_window(time_tr_start, w12a, w12b, args, model, device, train_loader,
                     test_loader, init_lr*0.1, train_acc_array, test_acc_array)
        train_window(time_tr_start, w12b, w123a, args, model, device, train_loader,
                     test_loader, init_lr*0.05, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123a, w123b, args, model, device, train_loader,
                     test_loader, init_lr*0.01, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123b, w123c, args, model, device, train_loader,
                     test_loader, init_lr*0.005, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123c, w123d, args, model, device, train_loader,
                     test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    if w12a <= args.k_allTrain_epochs < w12b:
        # train_window(time_tr_start,0, w1, args, model, device, train_loader,
        #              test_loader, init_lr, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w1, w12a, args, model, device, train_loader,
        #              test_loader, init_lr*0.5, train_acc_array, test_acc_array)
        weight_fixed(time_tr_start, w12a, w12b, args, model, device, train_loader,
                     test_loader, init_lr*0.1, train_acc_array, test_acc_array)
        train_window(time_tr_start, w12b, w123a, args, model, device, train_loader,
                     test_loader, init_lr*0.05, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123a, w123b, args, model, device, train_loader,
                     test_loader, init_lr*0.01, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123b, w123c, args, model, device, train_loader,
                     test_loader, init_lr*0.005, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123c, w123d, args, model, device, train_loader,
                     test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    # lr windows 3
    if w12b <= args.k_allTrain_epochs < w123a:
        # train_window(time_tr_start,0, w1, args, model, device, train_loader,
        #              test_loader, init_lr, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w1, w12a, args, model, device, train_loader,
        #              test_loader, init_lr*0.5, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w12a, w12b, args, model, device, train_loader,
        #              test_loader, init_lr*0.1, train_acc_array, test_acc_array)
        weight_fixed(time_tr_start, w12b, w123a, args, model, device, train_loader,
                     test_loader, init_lr*0.05, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123a, w123b, args, model, device, train_loader,
                     test_loader, init_lr*0.01, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123b, w123c, args, model, device, train_loader,
                     test_loader, init_lr*0.005, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123c, w123d, args, model, device, train_loader,
                     test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    if w123a <= args.k_allTrain_epochs < w123b:
        # train_window(time_tr_start,0, w1, args, model, device, train_loader,
        #              test_loader, init_lr, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w1, w12a, args, model, device, train_loader,
        #              test_loader, init_lr*0.5, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w12a, w12b, args, model, device, train_loader,
        #              test_loader, init_lr*0.1, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w12b, w123a, args, model, device, train_loader,
        #             test_loader, init_lr*0.05, train_acc_array, test_acc_array)
        weight_fixed(time_tr_start, w123a, w123b, args, model, device, train_loader,
                     test_loader, init_lr*0.01, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123b, w123c, args, model, device, train_loader,
                     test_loader, init_lr*0.005, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123c, w123d, args, model, device, train_loader,
                     test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    if w123b <= args.k_allTrain_epochs < w123c:
        # train_window(time_tr_start,0, w1, args, model, device, train_loader,
        #              test_loader, init_lr, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w1, w12a, args, model, device, train_loader,
        #              test_loader, init_lr*0.5, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w12a, w12b, args, model, device, train_loader,
        #              test_loader, init_lr*0.1, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w12b, w123a, args, model, device, train_loader,
        #             test_loader, init_lr*0.05, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w123a, w123b, args, model, device, train_loader,
        #              test_loader, init_lr*0.01, train_acc_array, test_acc_array)
        weight_fixed(time_tr_start, w123b, w123c, args, model, device, train_loader,
                     test_loader, init_lr*0.005, train_acc_array, test_acc_array)
        train_window(time_tr_start, w123c, w123d, args, model, device, train_loader,
                     test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    if w123c <= args.k_allTrain_epochs < w123d:
        # train_window(time_tr_start,0, w1, args, model, device, train_loader,
        #              test_loader, init_lr, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w1, w12a, args, model, device, train_loader,
        #              test_loader, init_lr*0.5, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w12a, w12b, args, model, device, train_loader,
        #              test_loader, init_lr*0.1, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w12b, w123a, args, model, device, train_loader,
        #             test_loader, init_lr*0.05, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w123a, w123b, args, model, device, train_loader,
        #              test_loader, init_lr*0.01, train_acc_array, test_acc_array)
        # train_window(time_tr_start,w123b, w123c, args, model, device, train_loader,
        #              test_loader, init_lr*0.005, train_acc_array, test_acc_array)
        weight_fixed(time_tr_start, w123c, w123d, args, model, device, train_loader,
                     test_loader, init_lr*0.001, train_acc_array, test_acc_array)

    # dir_model_state = "../Model_State/"+str(model.__class__.__name__)+"_C"+str(args.n_kernel)+"F10"+"/"+str(
    #     model.__class__.__name__)+"_C"+str(args.n_kernel)+"F10"+"_"+str(args.k_allTrain_epochs)+".pkl"
    # torch.save(model.state_dict(), dir_model_state)

    return train_acc_array, test_acc_array


def conv_fixed( time_start, L_W, R_W, args, model, device, train_loader, test_loader, lr, train_acc_array, test_acc_array):
    '''
    fixed convolutional layer
    '''
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.0005, momentum=0.9)
    print(optimizer)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # load the pretrained model state

    k_model_name = args.model_name+"_"+str(args.k_allTrain_epochs)
    dir_model_state = "../Model_State/"+args.model_name+"/"+k_model_name+".pkl"
    model.load_state_dict(torch.load(dir_model_state))

    # load the pretrained result sets
    dir_result_npz = "../Result_npz/"+args.model_name + \
        "/Acc_"+str(args.total_epochs)+".npz"
    Acc_results = np.load(dir_result_npz)
    test_results, train_results = Acc_results["arr_0"], Acc_results["arr_1"]

    pre_test_acc = test_results[:args.k_allTrain_epochs]
    for acc in pre_test_acc:
        test_acc_array.append(acc)

    pre_train_acc = train_results[:args.k_allTrain_epochs]
    for acc in pre_train_acc:
        train_acc_array.append(acc)

    # fixed convolutional layer
 
    layer_id = 0
    for child in model.children():
        layer_id += 1
        print("layer Id: "+str(layer_id), child)
        for param in child.parameters():
            param.requires_grad = False
            # print(param)

    model.fc[0].weight.requires_grad = True
    model.fc[0].bias.requires_grad = True
    model.fc[2].weight.requires_grad = True
    model.fc[2].bias.requires_grad = True

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.0005, momentum=0.9)
    print("---------> fixed the convolutional layer <---------")
    for epoch in range(args.k_allTrain_epochs + 1, R_W + 1):
        _train_epoch(epoch, args, model, device, train_loader, optimizer)
        train_acc = _train_acc(epoch, model, device, train_loader)
        train_acc_array.append(train_acc)
        test_acc = _test_epoch(time_start,epoch, model, device, test_loader, args)
        test_acc_array.append(test_acc)


def weight_fixed(time_start, L_W, R_W, args, model, device, train_loader, test_loader, lr, train_acc_array, test_acc_array):
    '''
    fixed weight of all layers
    '''
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.0005, momentum=0.9)
    print(optimizer)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # load the pretrained model state

    k_model_name = args.model_name+"_"+str(args.k_allTrain_epochs)
    dir_model_state = "../Model_State/"+args.model_name+"/"+k_model_name+".pkl"
    model.load_state_dict(torch.load(dir_model_state))

    # load the pretrained result sets
    dir_result_npz = "../Result_npz/"+args.model_name + \
        "/Acc_"+str(args.total_epochs)+".npz"
    Acc_results = np.load(dir_result_npz)
    test_results, train_results = Acc_results["arr_0"], Acc_results["arr_1"]

    pre_test_acc = test_results[:args.k_allTrain_epochs]
    for acc in pre_test_acc:
        test_acc_array.append(acc)

    pre_train_acc = train_results[:args.k_allTrain_epochs]
    for acc in pre_train_acc:
        train_acc_array.append(acc)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # for epoch in range(L_W+1, args.k_allTrain_epochs + 1):
    #     _train_epoch(epoch, args, model, device, train_loader, optimizer)
    #     train_acc = _train_acc(epoch, model, device, train_loader)
    #     train_acc_array.append(train_acc)
    #     test_acc = _test_epoch(epoch, model, device, test_loader, args)
    #     test_acc_array.append(test_acc)

    layer_id = 0
    for child in model.children():
        layer_id += 1
        print("layer Id: "+str(layer_id), child)
        for param in child.parameters():
            param.requires_grad = False
            # print(param)

    model.conv1[0].bias.requires_grad = True
    model.conv2[0].bias.requires_grad = True
    model.fc[0].bias.requires_grad = True
    model.fc[2].bias.requires_grad = True


    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.0005, momentum=0.9)
    print("---------> fixed the weight of all layers <---------")
    for epoch in range(args.k_allTrain_epochs + 1, R_W + 1):
        _train_epoch(epoch, args, model, device, train_loader, optimizer)
        train_acc = _train_acc(epoch, model, device, train_loader)
        train_acc_array.append(train_acc)
        test_acc = _test_epoch(time_start,epoch, model, device, test_loader, args)
        test_acc_array.append(test_acc)


def train_window(time_start, L_W, R_W, args, model, device, train_loader, test_loader, lr, train_acc_array, test_acc_array):
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.0005, momentum=0.9)
    print("---------> Step into another training window <---------")
    print(optimizer)

    for epoch in range(L_W+1, R_W + 1):
        # training the model
        _train_epoch(epoch, args, model, device, train_loader, optimizer)
        if args.get_state:
            k_model_name = args.model_name+"_"+str(epoch)
            dir_model_state = "../Model_State/"+args.model_name+"/"+k_model_name+".pkl"
            torch.save(model.state_dict(), dir_model_state)
            print("Save the model state: "+k_model_name+".pkl")
        # save state of model
        train_acc = _train_acc(epoch, model, device, train_loader)
        train_acc_array.append(train_acc)
        test_acc = _test_epoch(time_start,epoch, model, device, test_loader, args)
        test_acc_array.append(test_acc)


def _train_epoch(epoch, args, model, device, data_loader, optimizer):
    model.train()
    pid = os.getpid()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_loss = loss.item()
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
        #         pid, epoch, batch_idx * len(data), len(data_loader.dataset),
        #         100. * batch_idx / len(data_loader), loss.item()))
    # return train_loss


def _train_acc(epoch, model, device, data_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target).sum().item()
    test_loss /= len(data_loader.dataset)
    print('-----------------------------------------------------------------')
    print('Train Epoch: {}, Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(epoch,
                                                                                               test_loss, correct, len(
                                                                                                   data_loader.dataset),
                                                                                               100. * correct / len(data_loader.dataset)))
    # with open("./Result_npz/outputlog.txt", "a+") as f:
    #     print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
    #     test_loss, correct, len(data_loader.dataset),
    #     100. * correct / len(data_loader.dataset)), file=f)
    acc = 100. * correct / len(data_loader.dataset)
    return acc


def _test_epoch(time_start,epoch, model, device, data_loader, args):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target).sum().item()
    test_loss /= len(data_loader.dataset)
    print('Train Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(epoch,
                                                                                              test_loss, correct, len(
                                                                                                  data_loader.dataset),
                                                                                              100. * correct / len(data_loader.dataset)))
    print('%s (%d %.2f%%) ' % (timeSince(time_start, epoch / args.total_epochs), epoch, epoch / args.total_epochs * 100))
    print('-----------------------------------------------------------------')
    with open("../Result_npz/"+args.model_name+"/TestLog_"+str(args.k_allTrain_epochs)+".txt", "a+") as f:
        print('Train Epoch: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(epoch,
                                                                                                  test_loss, correct, len(
                                                                                                      data_loader.dataset),
                                                                                                  100. * correct / len(data_loader.dataset)), file=f)
        print('%s (%d %.2f%%) ' % (timeSince(time_start, epoch / args.total_epochs), epoch, epoch / args.total_epochs * 100),file=f)
    return 100. * correct / len(data_loader.dataset)
