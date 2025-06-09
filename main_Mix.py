# coding=utf-8
import time
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import torch
import cv2
import argparse
import os

import SamMapperMix
from utils import get_Samples_GT, GT_To_One_Hot, evaluate_performance, compute_loss, Draw_Classification_Map

# if torch.cuda.is_available():
#     print("Computation on CUDA GPU device {}".format('1'))
#     device = torch.device('cuda:{}'.format('1'))
# else:
#     print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
#     device = torch.device('cpu')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_folder = 'model'
result_folder = 'results'

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

## Use self-defined hyper-parameters
# parser = argparse.ArgumentParser(description='args')
# parser.add_argument('--data', type=int, default=1) #dataset
# parser.add_argument('--tr', type=int, default=30) #train_samples_per_class

# parser.add_argument('--L1', type=int, default = 2) #number of stage
# parser.add_argument('--L2', type=int, default = 1) #number of mix block
# parser.add_argument('--H1', type=int, default = 2) #number of head of MSC layer 
# parser.add_argument('--H2', type=int, default = 2) #number of head of MSA layer
# parser.add_argument('--ratio', type=float, default = 1.0/3.0) #C_cnn/C_trans, [1/3, 1, 3]
# parser.add_argument('--scale', type=int, default=100) #train_samples_per_class
# args = parser.parse_args()
# for (FLAG, curr_train_ratio, stage_num, mix_block_num, msc_head_num, msa_head_num, channel_ratio, superpixel_scale) in \
#     [(args.data, args.tr, args.L1, args.L2, args.H1, args.H2, args.ratio, args.scale)]:

# [(1,30, 3,2,2,1, 1.0/3.0,100), (2,30, 3,1,4,4, 1.0/3.0,144), (3,30, 3,1,2,2, 3.0,100), (4,30, 2,2,2,2, 1.0/3.0,121)]:
## Use my hyper-parameters
for (FLAG, curr_train_ratio, stage_num, mix_block_num, msc_head_num, msa_head_num, channel_ratio, superpixel_scale) in \
    [(2,30, 3,1,4,4, 1.0/3.0,144)]:

    print("FLAG, curr_train_ratio, stage_num, mix_block_num, msc_head_num, msa_head_num, channel_ratio, superpixel_scale")
    print(FLAG, curr_train_ratio, stage_num, mix_block_num, msc_head_num, msa_head_num, channel_ratio, superpixel_scale)

    torch.cuda.empty_cache()
    OA_ALL = [];AA_ALL = [];KPP_ALL = [];AVG_ALL = [];Train_Time_ALL = [];Test_Time_ALL = []
    if curr_train_ratio < 1:
        samples_type = 'ratio' 
    else:
        samples_type = 'same_num'

    Seed_List = [111,222,333]#,444,555]
    
    if FLAG == 1:
        data_mat = sio.loadmat('data/shuangtai_hsi.mat')
        data = data_mat['shuangtai'][:,:,tuple([i for i in range(20,100)]+[i for i in range(112,142)]+[i for i in range(170,200)])]
        gt_mat = sio.loadmat('data/shuangtai_gt_1027.mat')
        gt = gt_mat['shuangtai_gt'][:data.shape[0],:data.shape[1]]
        class_count = 8  
        learning_rate = 5e-4  
        max_epoch = 150 
        dataset_name = "shuangtai_" 
    
    if FLAG == 2:
        data_mat = sio.loadmat('data/chongming_hsi.mat')
        data = data_mat['chongming'][:,:,tuple([i for i in range(20,100)]+[i for i in range(112,142)]+[i for i in range(170,200)])]
        gt = sio.loadmat('data/chongming_gt_1027.mat')["chongming_gt"][:data.shape[0],:data.shape[1]]      
        class_count = 9  
        learning_rate = 5e-4  
        max_epoch = 150 
        dataset_name = "chongming_" 

    if FLAG == 3: 
        data_mat= sio.loadmat('data/GF_YC_data.mat')
        data = data_mat['Data']
        gt_mat = sio.loadmat('data/GF_YC_gt.mat')
        gt = gt_mat['DataClass'][:data.shape[0],:data.shape[1]]
        class_count = 7 
        learning_rate = 5e-4  
        max_epoch = 100 
        dataset_name = "GF_YC_" 

    if FLAG == 4:
        data_mat = sio.loadmat('data/ZY_HHK_data108_20210929.mat')
        data = data_mat['Data']
        gt = sio.loadmat('data/ZY_HHK_gt108_20210929.mat')["DataClass"][:data.shape[0],:data.shape[1]]
        class_count = 8 
        learning_rate = 5e-4  
        max_epoch = 100 
        dataset_name = "ZY_HHK_2021_" 

    print(data.shape, np.unique(gt), "np.unique(gt)")
    path = 'results/' + dataset_name + 'results.txt'

    # if not os.path.exists(path):
    #     with open(path, 'w') as file:
    #         file.write('')  # 可写入初始内容，如文件头信息等

    for curr_seed in Seed_List:
        train_samples_gt, test_samples_gt = get_Samples_GT(curr_seed, gt, class_count, curr_train_ratio, 1, samples_type)
        ## 将数据预先padding成能被4整除的大小，借鉴simplecv
        # data, train_samples_gt, test_samples_gt = preset(data, train_samples_gt, test_samples_gt)
        Test_GT = test_samples_gt
        m, n, d = data.shape  
        height, width, bands = data.shape  
        ## gt → onehot
        train_samples_gt_onehot = GT_To_One_Hot(train_samples_gt, class_count)
        test_samples_gt_onehot = GT_To_One_Hot(test_samples_gt, class_count)
        train_samples_gt_onehot = np.reshape(train_samples_gt_onehot, [-1, class_count]).astype(int)
        test_samples_gt_onehot = np.reshape(test_samples_gt_onehot, [-1, class_count]).astype(int)  
        ## gt → mask
        train_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        train_samples_gt = np.reshape(train_samples_gt, [m * n])
        for i in range(m * n):
            if train_samples_gt[i] != 0:
                train_label_mask[i] = temp_ones
        train_label_mask = np.reshape(train_label_mask, [m* n, class_count])
        test_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        test_samples_gt = np.reshape(test_samples_gt, [m * n])
        for i in range(m * n):
            if test_samples_gt[i] != 0:
                test_label_mask[i] = temp_ones
        test_label_mask = np.reshape(test_label_mask, [m* n, class_count])
        ## HSI data归一化
        data = np.reshape(data, [height * width, bands])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        data = np.reshape(data, [height, width, bands])

        # # 打印每类样本个数
        gt_reshape=np.reshape(gt, [-1])
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            print(samplesCount)

        LDA_SLIC_Time = 0.0

        train_samples_gt=torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
        test_samples_gt=torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)

        train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
        test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)

        train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
        test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)

        net_input=np.array(data,np.float32)
        net_input=torch.from_numpy(net_input.astype(np.float32)).to(device)
        net = SamMapperMix.SamMapperMix(height, width, bands, class_count, stage_num, mix_block_num, msc_head_num, msa_head_num, channel_ratio, superpixel_scale) 

        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)#,weight_decay=0.0001

        # with torch.autograd.profiler.profile(use_cuda=True,profile_memory=True) as prof:
        #     output, vis = net(net_input)
        # print(prof)
        # prof.export_chrome_trace('profiles')
        # exit()
        
        ## train the network
        net.train()
        tic1 = time.clock()
        for i in range(max_epoch+1):
            optimizer.zero_grad()  # zero the gradient buffers
            output, vis = net(net_input)
            loss = compute_loss(output, train_samples_gt_onehot, train_label_mask, m, n)
            loss.backward(retain_graph=False)
            optimizer.step()  # Does the update
            if i%50==0:
                with torch.no_grad():
                    net.eval()
                    output, vis = net(net_input)
                    trainloss = compute_loss(output, train_samples_gt_onehot, train_label_mask, m, n)
                    trainOA = evaluate_performance(path, OA_ALL, AA_ALL, KPP_ALL, AVG_ALL, output, train_samples_gt, train_samples_gt_onehot, m, n, class_count, Test_GT)
                    valloss = compute_loss(output, test_samples_gt_onehot, test_label_mask, m, n)
                    valOA = evaluate_performance(path, OA_ALL, AA_ALL, KPP_ALL, AVG_ALL, output, test_samples_gt, test_samples_gt_onehot, m, n, class_count, Test_GT)
                    print("{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(i + 1), round(trainloss.item(),5), round(trainOA.item(),5), round(valloss.item(),5), round(valOA.item(),5)))
                    torch.save(net.state_dict(),"model/{}best_model.pt".format(dataset_name))
                        
                torch.cuda.empty_cache()
                net.train()
        toc1 = time.clock()
        print("\n\n====================training done. starting evaluation...========================\n")
        training_time=toc1 - tic1 + LDA_SLIC_Time 
        print("training_time", training_time)
        Train_Time_ALL.append(training_time)
        
        torch.cuda.empty_cache()
        with torch.no_grad():
            net.load_state_dict(torch.load("model/{}best_model.pt".format(dataset_name)))
            net.eval()
            tic2 = time.clock()
            output, vis = net(net_input)
            toc2 = time.clock()
            testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask, m, n)
            testOA, OA_ALL, AA_ALL, KPP_ALL, AVG_ALL = evaluate_performance(path, OA_ALL, AA_ALL, KPP_ALL, AVG_ALL, output, test_samples_gt, test_samples_gt_onehot, m, n, class_count, Test_GT, require_AA_KPP=True,printFlag=False)
            print(OA_ALL, AA_ALL, KPP_ALL)
            print("{}\ttest loss={}\t test OA={}".format(str(i + 1), testloss, testOA))
            #
            classification_map=torch.argmax(output, 1).reshape([height,width]).cpu()+1
            pred_mat=classification_map.data.numpy()
            sio.savemat("results/"+"{}SamMapperMix_pred_mat_{}.mat".format(dataset_name, testOA),{"pred_mat":pred_mat})
            Draw_Classification_Map(classification_map,"results/"+dataset_name+str(testOA)+str(testOA))
            testing_time=toc2 - tic2 + LDA_SLIC_Time 
            print("testing_time", testing_time)
            Test_Time_ALL.append(testing_time)

            # ## visualization 
            # x_visualize = vis.cpu().detach().numpy()
            # x_visualize = np.mean(x_visualize,axis=1).reshape(x_visualize.shape[-2],x_visualize.shape[-1])
            # x_visualize = (((x_visualize - np.min(x_visualize))/(np.max(x_visualize)-np.min(x_visualize)))*255).astype(np.uint8) #归一化并映射到0-255的整数，方便伪彩色化
            # x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)  # 伪彩色处理
            # cv2.imwrite('vis/{}mix_{}.jpg'.format(dataset_name, curr_seed), x_visualize) # 保存可视化图像
            
        torch.cuda.empty_cache()
        del net
            
    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    AVG_ALL = np.array(AVG_ALL)
    Train_Time_ALL=np.array(Train_Time_ALL)
    Test_Time_ALL=np.array(Test_Time_ALL)

    print("\nTrain ratio={}, SuperpixelScale={}".format(curr_train_ratio,superpixel_scale),
          "\n==============================================================================")
    print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
    print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
    print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
    print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
    print("Average training time:{}".format(np.mean(Train_Time_ALL)))
    print("Average testing time:{}".format(np.mean(Test_Time_ALL)))
    
    # save information
    f = open('results/' + dataset_name + 'results.txt', 'a+')
    str_results = '\n\n************************************************' \
    +"\nTrain ratio={}, SuperpixelScale={}".format(curr_train_ratio,superpixel_scale) \
    +'\nOA='+ str(np.mean(OA_ALL))+ '+-'+ str(np.std(OA_ALL)) \
    +'\nAA='+ str(np.mean(AA_ALL))+ '+-'+ str(np.std(AA_ALL)) \
    +'\nKpp='+ str(np.mean(KPP_ALL))+ '+-'+ str(np.std(KPP_ALL)) \
    +'\nAVG='+ str(np.mean(AVG_ALL,0))+ '+-'+ str(np.std(AVG_ALL,0)) \
    +"\nAverage training time:{}".format(np.mean(Train_Time_ALL)) \
    +"\nAverage testing time:{}".format(np.mean(Test_Time_ALL))
    f.write(str_results)
    f.close()
        

    
    
    
    
    
    
    
