import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *
from generator_predict import *
from PIL import Image
import torch
import tifffile
import numpy as np
from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = 1e11
import argparse
from models import *
from torchvision.transforms import Normalize

normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

parser = argparse.ArgumentParser(
    description="This code generate all the predictions/ground truth values/features of each patch of each slide of the original test set"
)
parser.add_argument(
    "-np",
    "--n_passes",
    help="n_passes is the number of predictions you want to generate from the monte carlo model",
    type=int,
    default=20,
)
parser.add_argument(
    "-m",
    "--model",
    help="choose either vgg16 or resnet50",
    type=str,
)

args = parser.parse_args()
name_model = args.model
n_passes = args.n_passes

def enable_dropout(model):
    """Function to enable the dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

if name_model =='vgg16':
    print('you chose to work with the {}'.format(name_model))
    model = VGG16(vgg16(pretrained=False)).cuda()
    model.load_state_dict(torch.load(os.path.join(path_weights,'weights_'+str(percentage_scribbled_regions))))
    
    class Monte_carlo_model(torch.nn.Module):
        def __init__(self, model, n_passes=n_passes):
            super(Monte_carlo_model, self).__init__()

            self.model = model
            self.n_passes = n_passes

        def forward(self, x):
            x_features = self.model.vgg16.features(x)
            x_features = self.model.vgg16.avgpool(x_features)
            x_features = torch.nn.Flatten(start_dim=1, end_dim=-1)(x_features)
            x_features = self.model.vgg16.classifier[0](x_features)
            predictions = []

            for i in range(self.n_passes):
                pred = self.model.vgg16.classifier[1:](x_features)  ## 4096
                pred = self.model.relu(pred)
                pred = self.model.fc(pred)  ## 1
                pred = self.model.sigmoid(pred)
                predictions.append(pred)

            predictions = torch.stack(predictions)
            return x_features, predictions

if name_model == 'resnet50': 
    print('you chose to work with the {}'.format(name_model))
    model = RESNET50(resnet50(pretrained=False)).cuda()
    model.load_state_dict(torch.load(os.path.join(path_weights,'weights_'+ name_model +str(percentage_scribbled_regions))))

    class Monte_carlo_model(torch.nn.Module):
        def __init__(self, model, n_passes=n_passes):
            super(Monte_carlo_model, self).__init__()

            self.model = model
            self.n_passes = n_passes

        def forward(self, x):
            x_features = self.model.resnet50(x)
            predictions = []
            for i in range(self.n_passes):
                pred = self.model.relu(x_features) 
                pred = self.model.d1(self.model.relu(self.model.fc1(pred)))
                pred = self.model.d2(self.model.relu(self.model.fc2(pred))) 
                pred = self.model.d3(self.model.relu(self.model.fc3(pred)))
                pred = self.model.fc4(pred)
                pred = self.model.sigmoid(pred)
                predictions.append(pred)
            predictions = torch.stack(predictions)
            return x_features, predictions
else:
    print('wrong model name choose either vgg16 of resnet50')   

mc_model = Monte_carlo_model(model=model, n_passes=n_passes)
mc_model.cuda()
def evaluate(model, val_dl):
    # --- EVALUATE ON VALIDATION SET -------------------------------------
    model.eval()
    model.cuda()
    enable_dropout(model)

    all_labels = []
    all_predicted_labels = []
    all_features = []
    mean = torch.zeros(1).cuda()
    with torch.no_grad():
        for batch in tqdm(val_dl):
            images = normalize(batch[0].float().cuda())
            ys = batch[1].float().cpu().detach().numpy()
            out = model(images)
            pred_ys = out[1].cpu().detach().numpy()
            features = out[0].cpu().detach().numpy()
            all_labels.append(ys)
            all_predicted_labels.append(pred_ys)
            all_features.append(features)
    return all_labels, all_predicted_labels, all_features


filenames = os.listdir(path_slide_tumor_test)

for filename in tqdm(filenames):
    print('processing image {} ....'.format(filename))
    # try:
        # # """ predicts the first heatmap values and features of the model """

    filename = filename.split(".")[0]
    rename_dir = os.path.join(path_patches_test, filename)
    dataset_test = CustomImageDataset(path_image = rename_dir)
    loader_test = DataLoader(
        batch_size = bs,
        dataset=dataset_test,
        num_workers = 32,
        shuffle=False)

    dataloaders = {'test':loader_test}

    _, predictions, all_features = evaluate(mc_model,dataloaders['test'])

    predictions_new = np.concatenate(predictions,axis = 1)
    all_features_new = np.concatenate(all_features)

    path_patches = os.path.join(path_patches_test, filename)

    path_pf = os.path.join(path_prediction_features, filename)

    if not os.path.exists(path_pf):
        os.makedirs(path_pf)

    path_mask = os.path.join(path_slide_true_masks, filename + ".tif")

    # # # """ saves the predictions and features of each patch of the wsi"""

    print('saving labels & features...')

    np.save(os.path.join(path_pf, 'predictions'+name_model+'.npy'), predictions_new)
    np.save(os.path.join(path_pf, 'features'+name_model+'.npy'), all_features_new)
    del predictions_new, all_features_new, predictions, all_features

# except:
#     pass
    print("an error has occured with the filename:{}".format(filename))