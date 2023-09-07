from config import *
from generator_predict import *





Image.MAX_IMAGE_PIXELS = 1e11

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


model.load_state_dict(torch.load(os.path.join(path_weights,'weights'+str(percentage_scribbled_regions))))
class Monte_carlo_model(torch.nn.Module):

    def __init__(self,
                 model,
                 n_passes = 20): 
        
        super(Monte_carlo_model, self).__init__()
        
        self.model = model
        self.n_passes = n_passes
        
    def forward(self, x):
        
        x_features = self.model.vgg16.features(x) 
        x_features = self.model.vgg16.avgpool(x_features)
        x_features = torch.nn.Flatten(start_dim=1, end_dim=-1)(x_features)
        x_features = self.model.vgg16.classifier[0](x_features)
        predictions  = []

é        for i in range(self.n_passes):
    
            pred = self.model.vgg16.classifier[1:](x_features)  ## 4096
            pred = self.model.relu(pred)
            pred = self.model.fc(pred) ## 1
            pred = self.model.sigmoid(pred)
            predictions.append(pred)

        predictions = torch.stack(predictions)
        return x_features, predictions
            
mc_model = Monte_carlo_model(model = model,
                             n_passes = n_passes)
mc_model.cuda()



def evaluate(model,
            val_dl):

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
            images    = batch[0].float().cuda()
            ys    = batch[1].float().cpu().detach().numpy()           
            out = model(images)
            pred_ys = out[1].cpu().detach().numpy()
            features = out[0].cpu().detach().numpy()
            all_labels.append(ys)
            all_predicted_labels.append(pred_ys)
            all_features.append(features)
    return all_labels, all_predicted_labels, all_features


filenames = os.listdir(path_slide_tumor_test)


for filename in tqdm(filenames[16:]):
    
    print('processing image {} ....'.format(filename))
    
    # # """ predicts the first heatmap values and features of the model """
    
    
    filename = filename[:-4]
    new_filename = filename.replace('_','')
    rename_dir = os.path.join(path_patches_test, filename)
    dataset_test = CustomImageDataset(path_image = rename_dir)   
    loader_test = DataLoader(
        batch_size = bs,
        dataset=dataset_test,
        num_workers = 16,
        shuffle=False)

    dataloaders = {'test':loader_test}
    
    _, predictions, all_features = evaluate(mc_model,dataloaders['test'])
    
    predictions_new = np.concatenate(predictions,axis = 1)
    all_features_new = np.concatenate(all_features)
    
    path_patches = os.path.join(path_patches_test,filename)

    path_pf = os.path.join(path_prediction_features,filename)

    if not os.path.exists(path_pf):
        os.makedirs(path_pf, 0o0777)

    path_mask= os.path.join(path_slide_true_masks,new_filename)
    mask_png = os.path.join(path_mask,os.listdir(path_mask)[0])

    # # """ saves the predictions and features of each patch of the image"""
    
    print('saving labels & features...')

    
    np.save(os.path.join(path_pf, 'predictions.npy'), predictions_new)
    np.save(os.path.join(path_pf, 'features.npy'), all_features_new)
    
    del predictions_new, all_features_new, predictions, all_features
    
    # """ calculates the ground truth values of each patch of the image 
    #     the ground truth gt = 1.e if over half of the patch covered by tumor mask """
    
    print('retrieving labels from mask ...')

    path_patches = os.path.join(path_patches_test,filename)
    path_pf = os.path.join(path_prediction_features,filename)
    path_mask= os.path.join(path_slide_true_masks,new_filename)
    mask_png = os.path.join(path_mask,os.listdir(path_mask)[0])

    img = Image.open(mask_png)
    img_arr = np.asarray(img)
    files = os.listdir(path_patches)
    true_vals = np.zeros(len(files))

    for i , filename in enumerate(files):

        split = filename.split('_')

        y = int(split[2])
        x = int(split[3].split('.')[0])

        patch = img_arr[x:x+ps,y:y+ps]

        mean_ = np.mean(patch)


        th = 0.1
        if mean_>th:
            true_vals[i] = 1
        else: 
            true_vals[i] = 0

    np.save(os.path.join(path_pf,'trues.npy'), np.array(true_vals))

    del true_vals
    del img_arr

    img.close()