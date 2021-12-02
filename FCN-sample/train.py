import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import LoadDataset
from Models import FCN
import cfg
from metrics import *


device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
num_class = cfg.DATASET[1]

Load_train = LoadDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
Load_val = LoadDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)

train_data = DataLoader(Load_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)
val_data = DataLoader(Load_val, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)


fcn = FCN.FCN(num_class)
fcn = fcn.to(device)
criterion = nn.NLLLoss().to(device)     # 没有做softmax的交叉熵函数,需要先做log_softmax,才相当于CrossEntropyLoss
optimizer = optim.Adam(fcn.parameters(), lr=1e-4)       # 以上基本都是一些固定的写法


def train(model):
    best = [0]
    train_loss = 0
    net = model.train()
    running_metrics_val = runningScore(12)      # 这里是评价指标

    # 训练轮次
    for epoch in range(cfg.EPOCH_NUMBER):
        running_metrics_val.reset()
        print('Epoch is [{}/{}]'.format(epoch + 1, cfg.EPOCH_NUMBER))
        if epoch % 50 == 0 and epoch != 0:      # 下降学习率
            for group in optimizer.param_groups:
                group['lr'] *= 0.5
        # 训练批次
        for i, sample in enumerate(train_data):
            # 载入数据
            img_data = Variable(sample['img'].to(device))       # 这里是必须的，加入variable这个壳才可以自动进行反向传播，求梯度等
            img_label = Variable(sample['label'].to(device))
            # 训练
            out = net(img_data)
            # Loss计算
            # 这里进行了softmax,再log一下，dim=1表示每一行，常用，如果用了NLLLoss就用log_softmax
            # dim=1：对每一行的所有元素进行运算，并使得每一行所有元素和为1。
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label)
            # 反向传播，一定要先把梯度清零。
            optimizer.zero_grad()      # 当网络参量进行反馈时，梯度是累积计算而不是被替换，但在处理每一个batch时并不需要与其他batch的梯度混合起来累积计算，因此需要对每个batch调用一遍zero_grad（）将参数梯度置0.
            loss.backward()     # 反向传播，计算当前梯度
            optimizer.step()    # 根据梯度更新网络参数
            train_loss += loss.item()       # item()的作用是取出一个tensor中的某个元素值，不对向量型的tensor起作用。至于data，则是一个深拷贝的作用

            # 评估
            pre_label = out.max(dim=1)[1].data.cpu().numpy()    #out.max()返回 对应维度的最大值和对应的索引，[1]取索引，.data 从variable中提出来tensor，.cpu都放到cpu上再转为numpy
            true_label = img_label.data.cpu().numpy()
            running_metrics_val.update(true_label, pre_label)
            print('|batch[{}/{}]|batch_loss {: .8f}|'.format(i + 1, len(train_data), loss.item()))

        print("*******Batch END**********")
        metrics = running_metrics_val.get_scores()
        for k, v in metrics[0].items():
            print(k, v)
        train_miou = metrics[0]['mIou: ']
        if max(best) <= train_miou:
            best.append(train_miou)
            t.save(net.state_dict(), './Results/weights/FCN_weight/camvid{}.pth'.format(epoch))


def evaluate(model):
    net = model.eval()
    running_metrics_val = runningScore(12)
    eval_loss = 0
    prec_time = datetime.now()

    for j, sample in enumerate(val_data):
        valImg = Variable(sample['img'].to(device))
        valLabel = Variable(sample['label'].long().to(device))

        out = net(valImg)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out, valLabel)
        eval_loss = loss.item() + eval_loss
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        true_label = valLabel.data.cpu().numpy()
        running_metrics_val.update(true_label, pre_label)
    metrics = running_metrics_val.get_scores()
    for k, v in metrics[0].items():
        print(k, v)

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(time_str)


if __name__ == "__main__":
    train(fcn)
