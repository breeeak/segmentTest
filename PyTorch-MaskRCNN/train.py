import bisect
import glob
import os
import re
import time
import torch
import pytorch_mask_rcnn as pmr     # 会先去__init__，找可以导入的模块。
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)     # cpu 种子
    torch.cuda.manual_seed_all(seed)    # gpu种子
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True   # cudnn



    
def main(args):
    setup_seed(777)     # 设置随机数种子
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")  # 如果可以用GPU就用GPU
    if device.type == "cuda":  # 输出每个GPU的基本信息，包括型号，显存等
        pmr.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))  # 输出是cuda还是cpu在运算
        
    # ---------------------- prepare data loader ------------------------------- #
    
    dataset_train = pmr.datasets(args.dataset, args.data_dir, "train", train=True)  # 根据所选的数据集类型来声明训练数据集, train 说明是获取train.txt,在ImageSet Segmentation下面
    # dataset_train_0 = dataset_train[0]
    indices = torch.randperm(len(dataset_train)).tolist()  # 生成与数据集长度一致的index随机序列
    # class torch.utils.data.Subset(dataset, indices)`: 获取指定一个索引序列对应的子数据集
    d_train = torch.utils.data.Subset(dataset_train, indices)

    # 根据所选的数据集类型来声明测试数据集
    # d_test = pmr.datasets(args.dataset, args.data_dir, "val", train=True) # set train=True for eval
        
    args.warmup_iters = max(1000, len(d_train))
    
    # -------------------------------------------------------------------------- #

    print(args)
    num_classes = len(d_train.dataset.classes) + 1 # including background class 模型的类别中要多加背景一类
    model = pmr.maskrcnn_resnet50(True, num_classes).to(device)  # 新建模型，并读入预训练好的模型， True表示加载预训练模型
    
    params = [p for p in model.parameters() if p.requires_grad]  # 获取所有需要更新的变量
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)  # 设置优化器为SGD
    lr_lambda = lambda x: 0.1 ** bisect.bisect([22, 26], x)  # 随着epoch的变化的lr值，相当于一个倍率。bisect 是一个二分算法，查找元素在数组中的位置，返回索引。 1,0.1,0.01，22个epoch之前都是1
    
    start_epoch = 0
    # --------------------------------加载上次的训练节点------------------------------------------ #
    prefix, ext = os.path.splitext(args.ckpt_path)  # 将ckpt的地址分为前缀和名称
    ckpts = glob.glob(prefix + "-*" + ext)  # 搜索所有匹配ext的ckpt的名称
    ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))  # 按照ckpt中的数值来顺序排序 './weights/maskrcnn_voc-2'
    if ckpts:
        checkpoint = torch.load(ckpts[-1], map_location=device) # load last checkpoint 读取最新的ckpt
        model.load_state_dict(checkpoint["model"])  # 读入模型参数
        optimizer.load_state_dict(checkpoint["optimizer"])  # 读入优化器参数
        start_epoch = checkpoint["epochs"]  # 读入训练到的epoch是哪一位
        del checkpoint
        torch.cuda.empty_cache()    # 清缓存

    since = time.time()
    print("\nalready trained: {} epochs; to {} epochs".format(start_epoch, args.epochs))  # 输出此时训练了多少，还差多少没训练
    
    # ------------------------------- train ------------------------------------ #
        
    for epoch in range(start_epoch, args.epochs):
        print("\nepoch: {}".format(epoch + 1))
            
        A = time.time()  # 统计`train_one_epoch`函数的运行时间
        args.lr_epoch = lr_lambda(epoch) * args.lr  # 计算了此epoch的lr
        print("lr_epoch: {:.4f}, factor: {:.4f}".format(args.lr_epoch, lr_lambda(epoch)))
        iter_train = pmr.train_one_epoch(model, optimizer, d_train, device, epoch, args)
        A = time.time() - A
        
        B = time.time()  # 统计`evaluate`函数的运行时间，相当于验证集，这里没有使用
        # eval_output, iter_eval = pmr.evaluate(model, d_test, device, args)
        B = time.time() - B

        trained_epoch = epoch + 1
        print("training: {:.2f} s, evaluation: {:.2f} s".format(A, B))  # 返回训练和评价用的时间
        #pmr.collect_gpu_info("maskrcnn", [1 / iter_train, 1 / iter_eval])
        # print(eval_output.get_AP())

        # pmr.save_ckpt(model, optimizer, trained_epoch, args.ckpt_path, eval_info=str(eval_output))  # 保存这一个epoch中训练出的模型
        pmr.save_ckpt(model, optimizer, trained_epoch, args.ckpt_path)      # 还可以继续传入字典格式参数，保存模型的其他参数
        # it will create many checkpoint files during training, so delete some.  删除过多的ckpt文件，仅保留一部分
        prefix, ext = os.path.splitext(args.ckpt_path)
        ckpts = glob.glob(prefix + "-*" + ext)
        ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        n = 5
        if len(ckpts) > n:
            for i in range(len(ckpts) - n):
                os.remove(ckpts[i])
                # os.system("rm {}".format(ckpts[i])) 这是Linux删除，Windows不行
        
    # -------------------------------------------------------------------------- #

    print("\ntotal time of this training: {:.2f} s".format(time.time() - since))
    if start_epoch < args.epochs:
        print("already trained: {} epochs\n".format(trained_epoch))
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", action="store_true", default=True)  # 是否使用cuda
    
    parser.add_argument("--dataset", default="voc", help="coco or voc")  # 设置所采用的数据集是什么
    parser.add_argument("--data-dir", default="E:\\1_dataset\\common_dataset\\VOC\\data\\VOC_directly\\VOC2012")  # voc数据集的路径
    # parser.add_argument("--data-dir", default="/media/atara/WDSSD/PartTime/ObjectDetection/pytorch-retinanet/coco/")  # coco数据集的路径
    parser.add_argument("--ckpt-path")  # 训练好的模型地址
    parser.add_argument("--results")
    
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument('--lr-steps', nargs="+", type=int, default=[22, 26])#学习率调整的迭代次数
    parser.add_argument("--lr", type=float)  # 学习率
    parser.add_argument("--momentum", type=float, default=0.9)  # 动量有多大
    parser.add_argument("--weight-decay", type=float, default=0.0001)  # 衰减
    
    parser.add_argument("--epochs", type=int, default=50)  # 训练的epoch数
    parser.add_argument("--iters", type=int, default=200, help="max iters per epoch, -1 denotes auto")  # 每一个epoch训练多少个iter
    parser.add_argument("--print-freq", type=int, default=100, help="frequency of printing losses")  # 输出训练进度的频率
    args = parser.parse_args()
    
    if args.lr is None:
        args.lr = 0.02 * 1 / 16 # lr should be 'batch_size / 16 * 0.02'
    if args.ckpt_path is None:
        args.ckpt_path = "./weights/maskrcnn_{}.pth".format(args.dataset)
    if args.results is None:
        args.results = os.path.join(os.path.dirname(args.ckpt_path), "results.pth")
    
    main(args)
