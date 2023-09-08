import sys
import pathlib
sys.path.append(pathlib.Path(__file__).parent.parent)
from config import *
from scribble_inside_shape import  *
import warnings
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from multiprocessing import Pool
from utils import *
import pandas as pd

split = 'test'

if split == 'train':
    path_slide= path_slide_tumor_train
    filenames = os.listdir(path_slide)
    path_dataframe = path_dataframe_train
else:
    path_slide= path_slide_tumor_test
    filenames = os.listdir(path_slide)
    path_dataframe = path_dataframe_test

dic = {} 


for i, filename in enumerate(tqdm(os.listdir(path_slide))):

    path_image = os.path.join(path_slide,filename)
    annotations_tumor, scribbles_tumor, annotation_healthy, scribble_healthy = get_scribbles_and_annotations(path_image,
                                                                                                                split)
    dic[filename] = [scribbles_tumor, scribble_healthy, annotations_tumor]
    

    ###### SELECT x% of tumor regions ######
    ###### Remove scribble Healthy on tumor regions ######
    if dic[filename][1] is not None:
        n = dic[filename][1].shape[0]
        bool_filename = np.ones(n)
        n_tumor = len(dic[filename][0])
        areas = np.zeros(n_tumor)

        for i in range(n):
            point  = Point(dic[filename][1][i])
            for j, polygon in enumerate(dic[filename][2]):
                poly = Polygon(np.squeeze(polygon))
                if i == 0:
                    areas[j] = poly.area
                if poly.contains(point):
                    bool_filename[i] = 0
                    break
        dic[filename][1] = dic[filename][1][bool_filename.astype(bool)]
        args = np.flip(np.argsort(areas))[:np.minimum(int(percentage_scribbled_regions * n_tumor) + 1,10)]
        dic[filename][0] = [dic[filename][0][u] for u in args]
    

df = pd.DataFrame(columns = ['wsi', 'point', 'class'])

for filename in tqdm(dic.keys()):
    scribble_tumor, scribble_healthy,_ = dic[filename]
    scribble_tumor = np.squeeze(np.concatenate(scribble_tumor))

    for i in range(scribble_tumor.shape[0]): 
        df = df.append({'wsi':filename, 'point':scribble_tumor[i],'class':1}, ignore_index=True)
    if scribble_healthy is not None:
        for i in range(scribble_healthy.shape[0]):
            df = df.append({'wsi':filename, 'point':scribble_healthy[i],'class':0}, ignore_index=True)

df.to_csv(path_dataframe) 