import bisect
import glob
import os
import re
import time
import wandb

import torch

import pytorch_mask_rcnn as pmr
    
    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if device.type == "cuda": 
        pmr.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))
        
    # ---------------------- prepare data loader ------------------------------- #
    
    dataset_train = pmr.datasets(args.dataset, args.data_dir, "train2017", train=True)
    indices = torch.randperm(len(dataset_train)).tolist()
    d_train = torch.utils.data.Subset(dataset_train, indices)
    
    d_test = pmr.datasets(args.dataset, args.data_dir, "val2017", train=True) # set train=True for eval
        
    args.warmup_iters = max(1000, len(d_train))
    
    # -------------------------------------------------------------------------- #

    print(args)
    num_classes = max(d_train.dataset.classes) + 1 # including background class
    if 'fpn' in args.backbone:
        backbone_name = re.findall('(.*?)_fpn', args.backbone)[0]
        model = pmr.maskrcnn_resnet_fpn(pretrained=False, num_classes=num_classes,
                                        pretrained_backbone=True, backbone_name=backbone_name).to(device)
    else:
        model = pmr.maskrcnn_resnet50(False, num_classes, pretrained_backbone=True).to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = None
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            betas=(0.9, 0.999),  # You need to define these parameters
            eps=1e-8,  # You need to define this parameter
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_lambda = lambda x: 0.1 ** bisect.bisect(args.lr_steps, x)
    
    run = wandb.init(
    # Set the project where this run will be logged
    project="maskrcnn-bone-level-segmentation",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "patience": args.patience,
    },
    )

    start_epoch = 0
    
    # find all checkpoints, and load the latest checkpoint
    prefix, ext = os.path.splitext(args.ckpt_path)
    ckpts = glob.glob(prefix + "-*" + ext)
    ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    if ckpts:
        checkpoint = torch.load(ckpts[-1], map_location=device) # load last checkpoint
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epochs"]
        del checkpoint
        torch.cuda.empty_cache()

    since = time.time()
    print("\nalready trained: {} epochs; to {} epochs".format(start_epoch, args.epochs))
    
    # ------------------------------- train ------------------------------------ #

    best_validation_ap = -1
    epochs_since_improvement = 0
    patience = args.patience  # Number of epochs to wait for improvement

    for epoch in range(start_epoch, args.epochs):
        print("\nepoch: {}".format(epoch + 1))
            
        A = time.time()
        args.lr_epoch = lr_lambda(epoch) * args.lr
        print("lr_epoch: {:.5f}, factor: {:.5f}".format(args.lr_epoch, lr_lambda(epoch)))
        iter_train = pmr.train_one_epoch(model, optimizer, d_train, device, epoch, args)
        A = time.time() - A
        
        B = time.time()
        train_output, __iter_train = pmr.evaluate(model, dataset_train, device, args)
        eval_output, iter_eval = pmr.evaluate(model, d_test, device, args)
        B = time.time() - B

        trained_epoch = epoch + 1
        print("training: {:.1f} s, evaluation: {:.1f} s".format(A, B))
        pmr.collect_gpu_info("maskrcnn", [1 / iter_train, 1 / iter_eval])

        results_train = train_output.get_AP()
        print(results_train)

        results = eval_output.get_AP()
        print(results)

        print(train_output)
        print(eval_output)

        wandb.log({
            "train": {"bbox AP": results_train["bbox AP"], "mask AP": results_train["mask AP"]},
            "validation": {"bbox AP": results["bbox AP"], "mask AP": results["mask AP"]}
            })

        pmr.save_ckpt(model, optimizer, trained_epoch, args.ckpt_path, eval_info=str(eval_output))

        # Check if validation performance has improved
        if results["bbox AP"] > best_validation_ap:
            best_validation_ap = results["bbox AP"]
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        # Early stopping check
        if epochs_since_improvement >= patience:
            print(f"No improvement in validation AP for {patience} epochs. Stopping training.")
            break

        # it will create many checkpoint files during training, so delete some.
        prefix, ext = os.path.splitext(args.ckpt_path)
        ckpts = glob.glob(prefix + "-*" + ext)
        ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        n = 10
        if len(ckpts) > n:
            for i in range(len(ckpts) - n):
                os.system("rm {}".format(ckpts[i]))
        
    # -------------------------------------------------------------------------- #

    print("\ntotal time of this training: {:.1f} s".format(time.time() - since))
    if start_epoch < args.epochs:
        print("already trained: {} epochs\n".format(trained_epoch))
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", action="store_true")
    
    parser.add_argument("--dataset", default="coco", help="coco or voc")
    parser.add_argument("--data-dir", default="E:/PyTorch/data/coco2017")
    parser.add_argument("--ckpt-path")
    parser.add_argument("--results")
    parser.add_argument("--backbone", type=str, default="resnet101_fpn", choices=["resnet50", "resnet50_fpn", "resnet101_fpn"])
    
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument('--lr-steps', nargs="+", type=int, default=[6, 7])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10, help="max iters per epoch, -1 denotes auto")
    parser.add_argument("--print-freq", type=int, default=100, help="frequency of printing losses")
    parser.add_argument("--optimizer", default="adam", help="adam or sgd")
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    
    if args.lr is None:
        args.lr = 0.02 * 1 / 16 # lr should be 'batch_size / 16 * 0.02'
    if args.ckpt_path is None:
        args.ckpt_path = "./maskrcnn_{}.pth".format(args.dataset)
    if args.results is None:
        args.results = os.path.join(os.path.dirname(args.ckpt_path), "maskrcnn_results.pth")
    
    main(args)
    
    