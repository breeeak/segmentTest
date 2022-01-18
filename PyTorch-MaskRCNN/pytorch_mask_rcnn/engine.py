import time

import torch
import sys

from .utils import Meter, TextArea
try:
    from .datasets import CocoEvaluator, prepare_for_coco
except:
    pass


def train_one_epoch(model, optimizer, data_loader, device, epoch, args):  # 等于将data_loader的数据全部跑一次
    for p in optimizer.param_groups:
        p["lr"] = args.lr_epoch

    iters = len(data_loader) if args.iters < 0 else args.iters

    t_m = Meter("total")  # Meter是一个类，用于统计total的值
    m_m = Meter("model")  # Meter是一个类，用于统计模型平均运行时间的值
    b_m = Meter("backward")  # Meter是一个类，用于每一个iter的平均运行时间的值
    model.train()  # 将模型设置为训练模式
    A = time.time()
    for i, (image, target) in enumerate(data_loader):  # 每次读入一个数据
        T = time.time()
        num_iters = epoch * len(data_loader) + i  # 计算此时的总iters为多少
        if num_iters <= args.warmup_iters:  # 用于更新learning rate
            r = num_iters / args.warmup_iters
            for j, p in enumerate(optimizer.param_groups):
                p["lr"] = r * args.lr_epoch
                   
        image = image.to(device)  # 将image和target转换成对应的数据类型
        target = {k: v.to(device) for k, v in target.items()}
        S = time.time()
        
        losses = model(image, target)  # 将image和target输入模型，获得结果
        total_loss = sum(losses.values())
        m_m.update(time.time() - S)  # 用于统计模型平均运行时间的值
            
        S = time.time()
        total_loss.backward()  # 计算导数
        optimizer.step()  # 更新模型的变量
        optimizer.zero_grad()  # 清空上一次计算的导数
        b_m.update(time.time() - S)

        if num_iters % args.print_freq == 0:
            print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))

        t_m.update(time.time() - T)  # 用于每一个iter的平均运行时间的值
        if i >= iters - 1:
            break
           
    A = time.time() - A  # 一个`train_one_epoch()`函数总的运行时间
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}, backward: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg,1000*b_m.avg))
    return A / iters
            

def evaluate(model, data_loader, device, args, generate=True):
    if generate:
        iter_eval = generate_results(model, data_loader, device, args)

    dataset = data_loader #
    iou_types = ["bbox", "segm"]
    coco_evaluator = CocoEvaluator(dataset.coco, iou_types)

    results = torch.load(args.results, map_location="cpu")

    S = time.time()
    coco_evaluator.accumulate(results)
    print("accumulate: {:.1f}s".format(time.time() - S))

    # collect outputs of buildin function print
    temp = sys.stdout
    sys.stdout = TextArea()

    coco_evaluator.summarize()

    output = sys.stdout
    sys.stdout = temp
        
    return output, iter_eval
    
    
# generate results file   
@torch.no_grad()   
def generate_results(model, data_loader, device, args):
    iters = len(data_loader) if args.iters < 0 else args.iters
    ann_labels = data_loader.ann_labels
        
    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    model.eval()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        S = time.time()
        torch.cuda.synchronize()
        output = model(image)
        m_m.update(time.time() - S)
        
        prediction = {target["image_id"].item(): {k: v.cpu() for k, v in output.items()}}
        coco_results.extend(prepare_for_coco(prediction, ann_labels))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
     
    A = time.time() - A 
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg))
    
    S = time.time()
    print("all gather: {:.1f}s".format(time.time() - S))
    torch.save(coco_results, args.results)
        
    return A / iters
    

