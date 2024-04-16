import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *
from generator import *
from torchmetrics.classification import BinaryPrecision, BinaryRecall
from PIL import ImageFile
from torchvision.transforms import Normalize
import torch
from models import *
import neptune
import argparse

normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
prec = BinaryPrecision().cuda()
rec = BinaryRecall().cuda()


parser = argparse.ArgumentParser(
    description="Code to train the model on the scribble dataset"
)
parser.add_argument(
    "-m",
    "--model",
    help="choose either vgg or resnet",
    type=str,
)
args = parser.parse_args()
name_model = args.model

if name_model == "vgg":
    model = VGG16(vgg16(pretrained=True)).cuda()
else:
    model = RESNET50(resnet50(pretrained=True)).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
loss = torch.nn.BCELoss(reduction="mean")

run = neptune.init_run(
    mode="offline",
    project=os.environ["NEPTUNE_WORKSPACE"],
    api_token=os.environ["NEPTUNE_API_TOKEN"],
)
run["config/optimizer"] = "Adam"


def train(model, optimizer, train_dl, val_dl, epochs=100, loss=loss):
    tmp = (torch.ones(1) * 1e15).cuda()
    for epoch in range(1, epochs + 1):
        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        model.cuda()
        loss_tot = 0.0
        for batch in train_dl:
            optimizer.zero_grad()
            images = normalize(batch[0].float().cuda())
            ys = torch.unsqueeze(batch[1], dim=-1).float().cuda()
            pred_ys = model(images)
            loss_ = loss(pred_ys, ys)
            pred_ys = torch.flatten(pred_ys)
            ys = torch.flatten(ys)
            precision, recall = prec(pred_ys, ys.to(torch.int8)), rec(
                pred_ys, ys.to(torch.int8)
            )
            # backward
            loss_.backward()
            optimizer.step()
            run["train/epoch/loss_tot"].log(loss_)
            run["train/epoch/precision"].log(precision)
            run["train/epoch/recall"].log(recall)

        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        val_loss_tot = 0.0
        num_val_correct = 0
        num_val_examples = 0

        mean = torch.zeros(1).cuda()
        with torch.no_grad():
            for batch in val_dl:
                optimizer.zero_grad()
                images = normalize(batch[0].float().cuda())
                ys = torch.unsqueeze(batch[1], dim=-1).float().cuda()

                pred_ys = model(images)
                val_loss = loss(pred_ys, ys)
                pred_ys = torch.flatten(pred_ys)
                ys = torch.flatten(ys)
                precision, recall = prec(pred_ys, ys.to(torch.int8)), rec(
                    pred_ys, ys.to(torch.int8)
                )
                mean += val_loss
                optimizer.step()
                run["test/epoch/loss"].log(val_loss)
                run["test/epoch/precision"].log(precision)
                run["test/epoch/recall"].log(recall)
            mean = torch.mean(mean)

            if torch.gt(tmp, mean):
                print("the val loss decreased: saving the model...")
                tmp = mean
                if not os.path.exists(path_weights):
                    os.makedirs(path_weights)

                torch.save(
                    model.state_dict(),
                    os.path.join(
                        path_weights,
                        "weights_resnet50" + str(percentage_scribbled_regions),
                    ),
                )
    return 0

train(
    model,
    optimizer,
    dataloaders["train"],
    dataloaders["test"],
    epochs=40,
    loss=loss,
)
