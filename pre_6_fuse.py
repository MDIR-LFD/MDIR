
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

    get_T_net = get_T_2()
    get_T_net.apply(init_weights)
    get_T_net.to(device)

    get_T_net_R = get_T_3()
    get_T_net_R.apply(init_weights)
    get_T_net_R.to(device)

    enhance_L_net = EnhanceNet_L()  # .cuda()
    enhance_L_net.apply(init_weights)
    enhance_L_net.to(device)

    enhance_R_net = EnhanceNet_R_1()
    enhance_R_net.apply(init_weights)
    enhance_R_net.to(device)



    #enhance_T_net = enhance_T_1(heads = 1, qk_dim = 24, mlp_dim = 72)  # .cuda()
    #enhance_T_net = EnhanceNet_R()
    # enhance_T_net.apply(init_weights)
    # enhance_T_net.to(device)
    #
    Fuse_net = Fuse()
    Fuse_net.apply(init_weights)
    Fuse_net.to(device)

    # decomloss_part = DecomLoss_part()
    # decomloss_part.to(device)

    #Loss_ssim = ssim_loss_1()
    #Loss_ssim.to(device)


    # enhance_l_net = torch.nn.L1Loss()
    # enhance_l_net.to(device)
    #
    # enhance_r_net = torch.nn.L1Loss()
    # enhance_r_net.to(device)

    enhance_f_net = torch.nn.L1Loss()
    enhance_f_net.to(device)
    enhance_f_compose = torch.nn.L1Loss()
    enhance_f_compose.to(device)
    feature_extractor_f = FeatureExtractor()
    feature_extractor_f.eval()
    feature_extractor_f.to(device)
    enhance_f_feature = torch.nn.L1Loss()
    enhance_f_feature.to(device)

    # enhance_struct = torch.nn.L1Loss()
    # enhance_struct.to(device)


    feature_extractor = FeatureExtractor()
    feature_extractor.eval()
    feature_extractor.to(device)



    # optimizer_B = torch.optim.Adam(enhance_L_net.parameters(), lr=0.0002)
    # scheduler_B = CosineAnnealingLR(optimizer_B, T_max=epochs, eta_min=0)



    # optimizer_D = torch.optim.Adam(enhance_T_net.parameters(), lr=0.0002)
    # scheduler_D = CosineAnnealingLR(optimizer_D, T_max=epochs, eta_min=0)
    #
    optimizer_E = torch.optim.Adam(Fuse_net.parameters(), lr=0.0002)
    scheduler_E = CosineAnnealingLR(optimizer_E, T_max=epochs, eta_min=0)

    # optimizer_F = torch.optim.Adam(get_T_net.parameters(), lr=0.0002)
    # scheduler_F = CosineAnnealingLR(optimizer_F, T_max=epochs, eta_min=0)

   # best_ssim = -1
    #best_psnr = -1
    #best_images = {}  # 存储最优图像分量的字典
    decom_net.load_state_dict(torch.load('/home/jiangyonglong/program/Project3/save_model/decom_net_epoch_276.pth'))
    enhance_L_net.load_state_dict(torch.load('/home/jiangyonglong/program/Project3/save_model/enhance_L_net_epoch_499.pth'))
    enhance_R_net.load_state_dict(torch.load('/home/jiangyonglong/program/Project3/save_model_2/enhance_T_net_epoch_499.pth'))
    get_T_net.load_state_dict(torch.load('/home/jiangyonglong/program/Project3/save_model/get_T_net_epoch_499.pth'))
    get_T_net_R.load_state_dict(torch.load('/home/jiangyonglong/program/Project3/save_model_2/get_TR_net_epoch_499.pth'))
    #decom_net.eval()
    total_start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()  # 开始记录单个epoch的运行时间
        decom_net.eval()
        enhance_L_net.eval()
        enhance_R_net.eval()
        Fuse_net.train()
        get_T_net.eval()
        get_T_net_R.eval()
        #optimizer_F.param_groups[0]['lr'] = lr[epoch] #分解
        #optimizer_E.param_groups[0]['lr'] = lr1[epoch] #学习
        for i, data0, in enumerate(train_loader):
            #optimizer_B.zero_grad()
            #optimizer_D.zero_grad()
            optimizer_E.zero_grad()
            #optimizer_F.zero_grad()



            high_im, low_im = data0
            high_im = high_im.to(device)
            low_im = low_im.to(device)

            # save_image(high_im,
            #            "/home/jiangyonglong/program/Project3/save_image_3/high_im_{}_{}.png".format(epoch, i),
            #            normalize=False)
            # save_image(low_im,
            #            "/home/jiangyonglong/program/Project3/save_image_3/low_im{}_{}.png".format(epoch, i),
            #            normalize=False)

            # 分解
            with torch.no_grad():
                high_r_part, high_l_part = decom_net(high_im)
                low_r_part, low_l_part = decom_net(low_im)

            t1 = get_T_net(low_im)
            t2 = get_T_net_R(low_im)

            new_l_part = enhance_L_net(t1, low_l_part)  #$t, low_l
            r = enhance_R_net(t2, low_l_part, low_r_part)

            # r = enhance_T_net(t2, low_l_part, low_r_part)   #t, low_l, low_r
            #
            # out = Fuse_net(r, new_l_part)

            #out = r + new_l_part
            out = Fuse_net(new_l_part, r)  #x, enhance_l, enhance_r

            # high_l_part_1 = torch.cat((high_l_part, high_l_part, high_l_part), dim=1)
            # loss_enhance_l = enhance_l_net(new_l_part, high_l_part_1)
            # Loss_l_smooth = grad_loss3_L(new_l_part, high_l_part_1)
            #
            # loss_B = loss_enhance_l + Loss_l_smooth

            #loss_enhance_r_smooth = pt_grad_loss(r, high_r_part)

            #loss_enhance_r_ssim = ssim_loss_1(r, high_r_part)

            #loss_enhance_r = enhance_r_net(r, high_r_part)
            #loss_C = loss_enhance_r_ssim + loss_enhance_r + loss_enhance_r_smooth

            #loss_E = enhance_struct(new_l_part * r, high_im)
            loss_enhance_f = enhance_f_net(out, high_im)
            loss_enhance_f_ssim = ssim_loss_1(out, high_im)
            features_f_output = feature_extractor_f(out)
            features_high_im = feature_extractor_f(high_im)
            loss_content_f = enhance_f_feature(features_f_output, features_high_im)
            #
            loss_D = loss_enhance_f + loss_enhance_f_ssim + loss_content_f


            #loss =  loss_D

            loss_D.backward()


            #torch.nn.utils.clip_grad_norm_(parameters=decom_net.parameters(), max_norm=0.9, norm_type=2)

            #optimizer_B.step()

            # optimizer_D.step()
            optimizer_E.step()
            #optimizer_F.step()

            sys.stdout.write(
                "\r[A_Net_Epoch %d/%d] [A_Net_Batch %d/%d] [A loss: %f]"
                % (epoch, epochs, i, len(train_loader), loss_D.item()))

            gc.collect()

        scheduler_E.step()

        # scheduler_D.step()
        # scheduler_E.step()
        #scheduler_F.step()
        epoch_end_time = time.time()  # 结束单个epoch的运行时间
        print(f"Epoch {epoch} 运行时间: {epoch_end_time - epoch_start_time:.2f}秒")



        decom_net.eval()
        enhance_L_net.eval()
        enhance_R_net.eval()
        Fuse_net.eval()
        get_T_net.eval()
        get_T_net_R.eval()

        with torch.no_grad():
            ssim_loss_total = 0.0
            psnr_loss_total = 0.0
            count = 0
            # current_epoch_images = {}  # 存储当前epoch的图像分量

            for a, data2, in enumerate(val_loader):
                test_start_time = time.time()  # 开始记录测试epoch的运行时间
                val_high_im, val_low_im = data2
                val_low_im = val_low_im.to(device)
                val_high_im = val_high_im.to(device)
                val_low_r_part, val_low_l_part = decom_net(val_low_im)
                val_high_r_part, val_high_l_part = decom_net(val_high_im)



                val_t1 = get_T_net(val_low_im)
                val_t2 = get_T_net_R(val_low_im)

                val_new_l_part = enhance_L_net(val_t1, val_low_l_part) #$t, low_l
                #r = enhance_R_net(t2, low_l_part, low_r_part)


                val_r = enhance_R_net(val_t2, val_low_l_part, val_low_r_part) #t, low_l, low_r

                val_out = Fuse_net(val_new_l_part, val_r) #x, enhance_l, enhance_r
                #val_out = Fuse_net(val_r, val_new_l_part)
                #val_out = val_new_l_part + val_r


                test_end_time = time.time()  # 结束测试epoch的运行时间

                sys.stdout.write(f"\r测试Epoch {epoch} 第{a}张 运行时间: {test_end_time - test_start_time:.2f}秒")
                sys.stdout.flush()



                ssim_val = ssim_1(val_out, val_high_im, data_range=1.).item()
                psnr_val = psnr_1(val_out, val_high_im, data_range=1.).item()
                #
                ssim_loss_total += ssim_val
                psnr_loss_total += psnr_val
                count += 1

                # 存储当前epoch的图像分量
                if epoch == 0 or epoch % 2 == 0 or epoch == epochs - 1:
                    torch.save(Fuse_net.state_dict(),
                               f'/home/jiangyonglong/program/Project3/save_model_3/Fuse_net_epoch_{epoch}.pth')
                    # torch.save(enhance_L_net.state_dict(),
                    #            f'/home/jiangyonglong/program/Project3/save_model/enhance_L_net_epoch_{epoch}.pth')
                    # torch.save(enhance_T_net.state_dict(),
                    #            f'/home/jiangyonglong/program/Project3/save_model/enhance_T_net_epoch_{epoch}.pth')



                    # save_image(val_r,
                    #            "/home/jiangyonglong/program/Project3/save_image/r_{}_{}.png".format(epoch, a),
                    #            normalize=False)
                    save_image(val_out,
                               "/home/jiangyonglong/program/Project3/save_image_3/out_{}_{}.png".format(epoch, a),
                               normalize=False)

                    # save_image(val_new_l_part,
                    #            "/home/jiangyonglong/program/Project3/save_image/val_new_l_{}_{}.png".format(epoch, a),
                    #            normalize=False)
                    # save_image(val_t,
                    #            "/home/jiangyonglong/program/Project3/save_image/val_t_{}_{}.png".format(epoch, a),
                    #            normalize=False)
                    # torch.save(enhance_L_net.state_dict(),
                    #            '/home/jiangyonglong/program/Project3/save_model/enhance_L_net.pth')
                    # torch.save(enhance_T_net.state_dict(),
                    #            '/home/jiangyonglong/program/Project3/save_model/enhance_T_net.pth')
                if epoch == 0:
                    save_image(val_low_r_part,
                               "/home/jiangyonglong/program/Project3/save_image_3/val_low_r_{}_{}.png".format(epoch, a),
                               normalize=False)
                    save_image(val_low_l_part,
                               "/home/jiangyonglong/program/Project3/save_image_3/val_low_l_{}_{}.png".format(epoch, a),
                               normalize=False)
                    save_image(val_high_r_part,
                               "/home/jiangyonglong/program/Project3/save_image_3/val_high_r_{}_{}.png".format(epoch, a),
                               normalize=False)
                    save_image(val_high_l_part,
                               "/home/jiangyonglong/program/Project3/save_image_3/val_high_l_{}_{}.png".format(epoch, a),
                               normalize=False)
                    save_image(val_new_l_part,
                                "/home/jiangyonglong/program/Project3/save_image_3/val_new_l_{}_{}.png".format(epoch, a),
                                normalize=False)
                    save_image(val_r,
                               "/home/jiangyonglong/program/Project3/save_image_3/val_r_{}_{}.png".format(epoch, a),
                               normalize=False)
                    save_image(val_t1,
                               "/home/jiangyonglong/program/Project3/save_image_3/val_t1_{}_{}.png".format(epoch, a),
                               normalize=False)
                    save_image(val_t2,
                               "/home/jiangyonglong/program/Project3/save_image_3/val_t2_{}_{}.png".format(epoch, a),
                               normalize=False)

                gc.collect()


            avg_ssim_loss = ssim_loss_total / count
            avg_psnr_loss = psnr_loss_total / count
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



            sys.stdout.write("[Avg SSIM: %.4f] [Avg PSNR: %.4f]\n" % (avg_ssim_loss, avg_psnr_loss))
            sys.stdout.flush()


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
                                     transforms.CenterCrop(size=(256, 256)),
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