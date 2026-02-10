
#from torchvision import transforms

#这是光照增强网络

#from utiles import *
from Dataset import *
from model import *
from Loss import *
from torch.autograd import Variable
import sys
import gc
from torchvision.utils import save_image
import time

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from piq import ssim as ssim_1, psnr as psnr_1

#device0 = 'cuda:0'
device = 'cuda:0'
path1 = "/home/jiangyonglong/program/object1/NH-HAZE"
path2 = ""

def decomposenet(train_loader, val_loader, device, epochs):
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    decom_net = DecomposeNet_1()  # .cuda()
    decom_net.apply(init_weights)
    decom_net.to(device)


    enhance_L_net = EnhanceNet_L_ablation()  # .cuda()
    enhance_L_net.apply(init_weights)
    enhance_L_net.to(device)




    enhance_l_net = torch.nn.L1Loss()
    enhance_l_net.to(device)
   



    optimizer_B = torch.optim.Adam(enhance_L_net.parameters(), lr=0.0002)
    scheduler_B = CosineAnnealingLR(optimizer_B, T_max=epochs, eta_min=0)



   
    decom_net.load_state_dict(torch.load('/home/jiangyonglong/program/Project3/save_model/decom_net_epoch_498.pth'))
    decom_net.eval()
    total_start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()  # 开始记录单个epoch的运行时间
        decom_net.eval()
        enhance_L_net.train()
      
        for i, data0, in enumerate(train_loader):
            optimizer_B.zero_grad()
           



            high_im, low_im = data0
            high_im = high_im.to(device)
            low_im = low_im.to(device)

            # 分解
            with torch.no_grad():
                high_r_part, high_l_part = decom_net(high_im)
                low_r_part, low_l_part = decom_net(low_im)

            #t = get_T_net(low_im)

            #new_l_part = enhance_L_net(t, low_l_part)  #$t, low_l
            new_l_part = enhance_L_net(low_l_part)

            # r = enhance_T_net(t2, low_l_part, low_r_part)   #t, low_l, low_r
            #
            # out = Fuse_net(r, new_l_part)

            #out = r + new_l_part
            #out = Fuse_net(low_im, new_l_part, r)  #x, enhance_l, enhance_r

            high_l_part_1 = torch.cat((high_l_part, high_l_part, high_l_part), dim=1)
            loss_enhance_l = enhance_l_net(new_l_part, high_l_part_1)
            #loss_enhance_l1 = enhance_l_net1(new_l_part, high_l_part_1)
            #Loss_l_smooth = grad_loss3_L(new_l_part, high_l_part_1)
            Loss_l_smooth = grad_loss4_L(new_l_part, high_l_part_1)

            loss_B = loss_enhance_l +  Loss_l_smooth

  

            loss_B.backward()


            optimizer_B.step()

       

            sys.stdout.write(
                "\r[A_Net_Epoch %d/%d] [A_Net_Batch %d/%d] [A loss: %f]"
                % (epoch, epochs, i, len(train_loader), loss_B.item()))

            gc.collect()

        scheduler_B.step()


        epoch_end_time = time.time()  # 结束单个epoch的运行时间
        print(f"Epoch {epoch} 运行时间: {epoch_end_time - epoch_start_time:.2f}秒")



        decom_net.eval()
        enhance_L_net.eval()
 

        with torch.no_grad():
        

            for a, data2, in enumerate(val_loader):
                test_start_time = time.time()  # 开始记录测试epoch的运行时间
           

                test_end_time = time.time()  # 结束测试epoch的运行时间

                sys.stdout.write(f"\r测试Epoch {epoch} 第{a}张 运行时间: {test_end_time - test_start_time:.2f}秒")
                sys.stdout.flush()



               

         
                if epoch == epochs - 1:
                    # torch.save(get_T_net.state_dict(),
                    #            f'/home/jiangyonglong/program/Project3/ablation/get_T_net_epoch_{epoch}.pth')
                    torch.save(enhance_L_net.state_dict(),
                               f'/home/jiangyonglong/program/Project3/ablation/enhance_L_net_epoch_{epoch}.pth')
              

                gc.collect()




    total_end_time = time.time()  # 结束记录总运行时间
    print(f"所有Epoch总运行时间: {total_end_time - total_start_time:.2f}秒")







folder_path = "/home/jiangyonglong/program/Project3"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device))
    #train_high_names, train_low_names, \
    #val_high_names, val_low_names = read_split_data(root4, root5)
    train_GT, train_hazy, val_GT, val_hazy= get_paired_image_paths_2(folder_path)

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     #transforms.CenterCrop(size=(256, 256)),
                                     #transforms.RandomResizedCrop(128),  # 随机裁剪
                                     #transforms.RandomHorizontalFlip(),  # 随机水平翻转
                                     #transforms.RandomVerticalFlip(),#随机上下翻转
                                       ##transforms.ToTensor()自动换通道顺序，不用手工转化
                                     # transforms.Resize(100),
                                     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     ]),
        "val": transforms.Compose([  # 最小边到256
                                      # 中心裁剪
                                    #transforms.RandomResizedCrop(96, 96),  # 随机裁剪
                                    #transforms.RandomHorizontalFlip(),  # 随机水平翻转
                                    transforms.ToTensor(),
                                    transforms.CenterCrop(size=(768, 768)),
                                    # transforms.Resize(100), # 最小边到256
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    train_data_set = CUTDataset_256_5(high_names=train_GT,
                                   low_names=train_hazy,
                                   transform=data_transform["train"])

    val_data_set = Mydataset_val(high_names=val_GT,
                                low_names=val_hazy,
                                transform=data_transform["val"])

    batch_size1 = 4


    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size1, shuffle=True, num_workers=5,
                                                collate_fn=train_data_set.collate_fn, pin_memory=True)
    val_batch_size = 1
    val_loader = torch.utils.data.DataLoader(val_data_set, val_batch_size, shuffle=False, num_workers=5,
                                             collate_fn=val_data_set.collate_fn, pin_memory=True)

    epochs = 400
    #test_image(train_loader)


    decomposenet(train_loader, val_loader, 'cuda:0', epochs)


if __name__ == '__main__':
    main()