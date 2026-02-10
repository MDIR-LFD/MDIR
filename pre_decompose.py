
#from torchvision import transforms


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
from piq import ssim, psnr

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
   

    decomloss_part = DecomLoss_part()
    decomloss_part.to(device)



    optimizer_A = torch.optim.Adam(decom_net.parameters(), lr=0.0001)
    scheduler_A = CosineAnnealingLR(optimizer_A, T_max=epochs, eta_min=0)
   
    # loss_aa = 100
    # best_images = {}  # 存储最优图像分量的字典
    total_start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()  # 开始记录单个epoch的运行时间
        #optimizer_F.param_groups[0]['lr'] = lr[epoch] #分解
        #optimizer_E.param_groups[0]['lr'] = lr1[epoch] #学习
        for i, data0, in enumerate(train_loader):
            decom_net.train()
        
            optimizer_A.zero_grad()
          


            high_im, low_im = data0
            high_im = high_im.to(device)
            low_im = low_im.to(device)
            decom_net.train()
            # 分解
            high_r_part, high_l_part = decom_net(high_im)
            low_r_part, low_l_part = decom_net(low_im)

           



            low_im = low_im.to(device)
            high_im = high_im.to(device)
            low_r_part = low_r_part.to(device)
            low_l_part = low_l_part.to(device)
            high_r_part = high_r_part.to(device)
            high_l_part = high_l_part.to(device)


            low_l_part1 = torch.cat((low_l_part, low_l_part, low_l_part), dim=1)
            high_l_part1 = torch.cat((high_l_part, high_l_part, high_l_part), dim=1)



            loss_decompose_net_low_cons = torch.mean(torch.abs((low_r_part * low_l_part1) - low_im))
            loss_decompose_net_high_cons = torch.mean(torch.abs((high_r_part * high_l_part1) - high_im))
            loss_decompose_net_R = torch.mean(torch.abs(low_r_part - high_r_part))

            loss_recon_mutal_low = torch.mean(torch.abs(high_r_part * low_l_part1 - low_im))
            loss_recon_mutal_high = torch.mean(torch.abs(low_r_part * high_l_part1 - high_im))

         


            


            loss_A = loss_decompose_net_low_cons + loss_decompose_net_high_cons + loss_decompose_net_R \
                +   loss_recon_mutal_low +  loss_recon_mutal_high 
                


          

            loss = loss_A 
            loss.backward()


           
            optimizer_A.step()
           

            sys.stdout.write(
                "\r[A_Net_Epoch %d/%d] [A_Net_Batch %d/%d] [A loss: %f]"
                % (epoch, epochs, i, len(train_loader), loss_A.item()))

            gc.collect()
        scheduler_A.step()
        
        epoch_end_time = time.time()  # 结束单个epoch的运行时间
        print(f"Epoch {epoch} 运行时间: {epoch_end_time - epoch_start_time:.2f}秒")



        decom_net.eval()

        with torch.no_grad():
            if epoch == 0 or epoch % 2 == 0:

                torch.save(decom_net.state_dict(), f'/home/jiangyonglong/program/Project3/save_model/decom_net_epoch_{epoch}.pth')

            for a, data2, in enumerate(val_loader):
                test_start_time = time.time()  # 开始记录测试epoch的运行时间
                val_high_im, val_low_im = data2
                val_low_im = val_low_im.to(device)
                val_high_im = val_high_im.to(device)
                val_low_r_part, val_low_l_part = decom_net(val_low_im)
                val_high_r_part, val_high_l_part = decom_net(val_high_im)

                test_end_time = time.time()  # 结束测试epoch的运行时间

                sys.stdout.write(f"\r测试Epoch {epoch} 第{a}张 运行时间: {test_end_time - test_start_time:.2f}秒")
                sys.stdout.flush()

                if epoch == 0 or epoch % 20 == 0 or epoch == epochs -1:
                # 存储当前epoch的图像分量

                    save_dir = "/home/jiangyonglong/program/Project3/save_image_1/"  # 您希望保存图像的目录
                    save_path = os.path.join(save_dir, f"val_low_r_part_{epoch}_{a}.png")
                    save_image(val_low_r_part.cpu(), save_path, normalize=False)

                    save_path = os.path.join(save_dir, f"val_low_l_part_{epoch}_{a}.png")
                    save_image(val_low_l_part.cpu(), save_path, normalize=False)

                    save_path = os.path.join(save_dir, f"val_high_r_part_{epoch}_{a}.png")
                    save_image(val_high_r_part.cpu(), save_path, normalize=False)

                    save_path = os.path.join(save_dir, f"val_high_l_part_{epoch}_{a}.png")
                    save_image(val_high_l_part.cpu(), save_path, normalize=False)

                # 保存模型参数

                      # 保存最佳模型下的图像



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
                                    transforms.CenterCrop(size=(576, 576)),
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