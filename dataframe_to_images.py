from config import *
from PIL import ImageFile
from multiprocessing import Pool
import ast

ImageFile.LOAD_TRUNCATED_IMAGES = True


split = 'test'

if split == 'train' : 
    path_patches = path_patches_scribbles_train
    path_images = path_slide_tumor_train
    path_dataframe = path_dataframe_train

else:
    path_patches = path_patches_scribbles_test
    path_images = path_slide_tumor_test
    path_dataframe = path_dataframe_test


dataframe = pd.read_csv(path_dataframe, index_col=0)

# #### CREATE DATASET FROM slides

slides = list(np.unique(np.array(list(dataframe['wsi']))))

def df_to_images(filename):
    
    dataframe_subset = dataframe[dataframe['wsi'] == filename]
    dataframe_subset = dataframe_subset.reset_index(drop=True)
    path = os.path.join(path_images,filename)
    img = OpenSlide(path)
    
    for idx in tqdm(range(dataframe_subset.shape[0])):
        filename, point, y = dataframe_subset.loc[idx]
        y = y.astype(int)
        point = ' '.join(point.split())
        point = point.replace('[ ','[')
        point = point.replace(' ', ',')
        point = tuple(np.array(ast.literal_eval(point)).astype(int) - ps//2)
        region  = img.read_region(point, level = 0, size = (ps,ps))
        region = region.convert("RGB")
        region.save(os.path.join(path_patches,'image_'+str(idx)+'_'+filename.split('_')[1] +'_'+str(y)+'.png'))
                
if __name__ == '__main__':
    
    pool = Pool(processes=16)                         
    pool.map(df_to_images, slides)

