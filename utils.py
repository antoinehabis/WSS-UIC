from config import *
from scribble_inside_shape import Scribble
from histolab.slide import Slide
from histolab.masks import TissueMask
from histolab.filters.image_filters import (
   ApplyMaskImage,
   GreenPenFilter,
   Invert,
   OtsuThreshold,
   RgbToGrayscale,
)
from histolab.filters.morphological_filters import RemoveSmallHoles, RemoveSmallObjects

def metrics(predictions, trues):

    tp = np.sum((predictions) * trues)
    tn = np.sum((1 - predictions) * (1 - trues))
    fp = np.sum((predictions) * (1 - trues))
    fn = np.sum((1 - predictions) * trues)
    eps = 1e-6
    accuracy = np.sum(trues  == (predictions))/trues.shape[0]
    precision_healthy = tn/ (tn + fn + eps)
    precision_tumor = tp / (tp + fp + eps)
    m_precision = (precision_healthy+precision_tumor)/2
    recall_tumor = tp / (tp + fn +eps)
    recall_healthy = tn / (tn + fp +eps)
    m_recall = (recall_tumor+recall_healthy)/2
    f1 = 2 * (m_precision * m_recall)/(m_precision + m_recall+eps)
    balanced_accuracy = (m_precision +m_recall)/2

    return balanced_accuracy, m_precision, m_recall, f1



def get_scribbles_and_annotations(path_image,
                                  split):
    filename = path_image.split('/')[-1]
    slide = Slide(path_image, processed_path='')
    
    mask = TissueMask(
       RgbToGrayscale(),
       OtsuThreshold(),
       ApplyMaskImage(slide.thumbnail),
       GreenPenFilter(),
       RgbToGrayscale(),
       Invert(),
       OtsuThreshold(),
       RemoveSmallHoles(),
       RemoveSmallObjects(min_size =0, avoid_overmask=False )
    )
    sf = 4
    k = np.array(slide.locate_mask(mask, scale_factor=sf,outline='green'))
    k[k==128] = 0
    
    contours,_ = cv2.findContours(k[:,:,3],cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sizes = np.array([contours[i].shape[0] for i in range(len(contours))])
    r = np.argmax(sizes)
    
    annotation_healthy = contours[r] *sf 
    
    s = Scribble(filename.split('.')[0],
                 percent=1.,
                 show = False,
                 split = split)
    
    dataframe_annotation = s.create_dataframe_annotations()
    scribbles_tumor = []
    annotations_tumor = []
    
    for annotation_id in tqdm(list(dataframe_annotation.columns)):
        
        scribble_tumor, annotation = s.final_scribble(dataframe_annotation,annotation_id)
        
        if scribble_tumor is not(None):
            scribble_tumor = scribble_tumor[s.interpolation_method]
            scribble_tumor =np.expand_dims(scribble_tumor, axis = 1)
            annotation =np.expand_dims(annotation, axis = 1)
            annotations_tumor.append(annotation)
            scribbles_tumor.append(scribble_tumor)
    try :
        scribble_healthy = s.scribble_background(annotation_healthy.squeeze())[0][s.interpolation_method]
    except:
        scribble_healthy = None

    return (annotations_tumor ,
            scribbles_tumor,
            annotation_healthy,
            scribble_healthy)


