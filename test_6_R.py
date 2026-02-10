
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
from piq import ssim as ssim_1, psnr as psnr_1

#device0 = 'cuda:0'
device = 'cuda:0'
device1 = 'cuda:1'
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
    decom_net.to(device1)

    # get_T_net = get_T_3()
    # get_T_net.apply(init_weights)
    # get_T_net.to(device)

    #enhance_T_net = EnhanceNet_R_1()
    enhance_T_net = EnhanceNet_R_ablation()
    enhance_T_net.apply(init_weights)
    enhance_T_net.to(device1)

    # enhance_r_net = torch.nn.L1Loss()
    # enhance_r_net.to(device)




   # best_ssim = -1
    #best_psnr = -1
    #best_images = {}  # 存储最优图像分量的字典
    decom_net.load_state_dict(torch.load('/home/jiangyonglong/program/Project3/save_model/decom_net_epoch_498.pth'))
    #get_T_net.load_state_dict(torch.load('/home/jiangyonglong/program/Project3/save_model_2/get_TR_net_epoch_499.pth'))
    #enhance_T_net.load_state_dict(torch.load('/home/jiangyonglong/program/Project3/save_model_2/enhance_T_net_epoch_499.pth'))
    enhance_T_net.load_state_dict(
        torch.load('/home/jiangyonglong/program/Project3/ablation/enhance_T_net_epoch_499.pth'))


    decom_net.eval()
    #enhance_L_net.eval()
    enhance_T_net.eval()
    #Fuse_net.eval()
    #get_T_net.eval()

    with torch.no_grad():
        # ssim_loss_total = 0.0
        # psnr_loss_total = 0.0
        # count = 0
        # current_epoch_images = {}  # 存储当前epoch的图像分量

        for a, data2, in enumerate(train_loader):
            test_start_time = time.time()  # 开始记录测试epoch的运行时间
            val_high_im, val_low_im = data2
            val_low_im = val_low_im.to(device1)
            #val_high_im = val_high_im.to(device)
            val_low_r_part, val_low_l_part = decom_net(val_low_im)
            #val_high_r_part, val_high_l_part = decom_net(val_high_im)
            val_low_im = val_low_im.to(device)

            #val_t = get_T_net(val_low_im)

            #val_new_l_part = enhance_L_net(val_t1, val_low_l_part) #$t, low_l
            #val_t = val_t.to(device1)


            #val_r = enhance_T_net(val_t, val_low_l_part, val_low_r_part) #t, low_l, low_r
            val_r = enhance_T_net(val_low_l_part, val_low_r_part)

            #val_out = Fuse_net(val_low_im, val_new_l_part, val_r) #x, enhance_l, enhance_r
            #val_out = Fuse_net(val_r, val_new_l_part)
            #val_out = val_new_l_part + val_r


            test_end_time = time.time()  # 结束测试epoch的运行时间

            sys.stdout.write(f"\r测试Epoch {0} 第{a}张 运行时间: {test_end_time - test_start_time:.2f}秒")
            sys.stdout.flush()



            # ssim_val = ssim_1(val_out, val_high_im, data_range=1.).item()
            # psnr_val = psnr_1(val_out, val_high_im, data_range=1.).item()
            #
            # ssim_loss_total += ssim_val
            # psnr_loss_total += psnr_val
            # count += 1

            # 存储当前epoch的图像分量





            # save_image(val_r,
            #            "/home/jiangyonglong/program/Project3/test_R_image/r_{}_{}.png".format(0, a),
            #            normalize=False)
            save_image(val_r,
                       "/home/jiangyonglong/program/Project3/test_R_image_ablation/r_{}_{}.png".format(0, a),
                       normalize=False)
            # save_image(val_t,
            #            "/home/jiangyonglong/program/Project3/test_Rt_image/val_tR_{}_{}.png".format(0, a),
            #            normalize=False)
                # save_image(val_out,
                #            "/home/jiangyonglong/program/Project3/save_image/out_{}_{}.png".format(epoch, a),
                #            normalize=False)
                #
                # save_image(val_new_l_part,
                #            "/home/jiangyonglong/program/Project3/save_image/val_new_l_{}_{}.png".format(epoch, a),
                #            normalize=False)
                # torch.save(enhance_L_net.state_dict(),
                #            '/home/jiangyonglong/program/Project3/save_model/enhance_L_net.pth')
                # torch.save(enhance_T_net.state_dict(),
                #            '/home/jiangyonglong/program/Project3/save_model/enhance_T_net.pth')
            # if epoch == 0:
            #     save_image(val_low_r_part,
            #                "/home/jiangyonglong/program/Project3/save_image/val_low_r_{}_{}.png".format(epoch, a),
            #                normalize=False)
            #     save_image(val_low_l_part,
            #                "/home/jiangyonglong/program/Project3/save_image/val_low_l_{}_{}.png".format(epoch, a),
            #                normalize=False)
            #     save_image(val_high_r_part,
            #                "/home/jiangyonglong/program/Project3/save_image/val_high_r_{}_{}.png".format(epoch, a),
            #                normalize=False)
            #     save_image(val_high_l_part,
            #                "/home/jiangyonglong/program/Project3/save_image/val_high_l_{}_{}.png".format(epoch, a),
            #                normalize=False)

            gc.collect()


            # avg_ssim_loss = ssim_loss_total / count
            # avg_psnr_loss = psnr_loss_total / count
            #
            # # 检查是否需要更新最佳图像和保存模型
            # if avg_ssim_loss > best_ssim:
            #     best_ssim = avg_ssim_loss


                # 保存模型参数
                #torch.save(decom_net.state_dict(), '/home/jiangyonglong/program/Project3/save_model/decom_net.pth')
                #torch.save(enhance_L_net.state_dict(), '/home/jiangyonglong/program/Project3/save_model/enhance_L_net.pth')
                #torch.save(enhance_R_net.state_dict(), '/home/jiangyonglong/program/Project3/save_model/enhance_R_net.pth')
                #torch.save(enhance_T_net.state_dict(), '/home/jiangyonglong/program/Project3/save_model/enhance_T_net.pth')
                # torch.save(Fuse_net.state_dict(),
                #            '/home/jiangyonglong/program/Project3/save_model/Fuse_net.pth')



            # sys.stdout.write("[Avg SSIM: %.4f] [Avg PSNR: %.4f]\n" % (avg_ssim_loss, avg_psnr_loss))
            # sys.stdout.flush()










folder_path = "/home/jiangyonglong/program/Project3"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device))
    #train_high_names, train_low_names, \
    #val_high_names, val_low_names = read_split_data(root4, root5)
    train_GT, train_hazy, val_GT, val_hazy= get_paired_image_paths_2(folder_path)

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.CenterCrop(size=(896, 1376)),
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

    train_data_set = CUTDataset_256_4(high_names=train_GT,
                                   low_names=train_hazy,
                                   transform=data_transform["train"])

    val_data_set = Mydataset_val(high_names=val_GT,
                                low_names=val_hazy,
                                transform=data_transform["val"])

    batch_size1 = 1


    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size1, shuffle=False, num_workers=5,
                                                collate_fn=train_data_set.collate_fn, pin_memory=True)
    val_batch_size = 1
    val_loader = torch.utils.data.DataLoader(val_data_set, val_batch_size, shuffle=False, num_workers=5,
                                             collate_fn=val_data_set.collate_fn, pin_memory=True)

    epochs = 500
    #test_image(train_loader)


    decomposenet(train_loader, val_loader, 'cuda:0', epochs)


if __name__ == '__main__':
    main()