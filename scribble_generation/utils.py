import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
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
import numpy as np
import cv2
from tqdm import tqdm


def get_scribbles_and_annotations(path_image, split):
    
    ### First extract the largest tissue component on the slide 
    #to generate a healthy scribble later

    filename = path_image.split("/")[-1]
    slide = Slide(path_image, processed_path="")

    mask = TissueMask(
        RgbToGrayscale(),
        OtsuThreshold(),
        ApplyMaskImage(slide.thumbnail),
        GreenPenFilter(),
        RgbToGrayscale(),
        Invert(),
        OtsuThreshold(),
        RemoveSmallHoles(),
        RemoveSmallObjects(min_size=0, avoid_overmask=False),
    )
    sf = 4
    k = np.array(slide.locate_mask(mask, scale_factor=sf, outline="green"))
    k[k == 128] = 0

    contours, _ = cv2.findContours(
        k[:, :, 3], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    sizes = np.array([contours[i].shape[0] for i in range(len(contours))])
    r = np.argmax(sizes)

    annotation_healthy = contours[r] * sf

    s = Scribble(filename.split(".")[0], percent=0.0, split=split)
    ### Extract the contours of all the tumor annotations
    dataframe_annotation = s.create_dataframe_annotations()

    ### For all the tumor annotations: generate a scribble inside the annotation
    scribbles_tumor = []
    annotations_tumor = []

    for annotation_id in tqdm(list(dataframe_annotation.columns)):
        annotation_contour = dataframe_annotation[annotation_id]
        annotation_contour = annotation_contour[~annotation_contour.isnull()]
        contour_tissue = np.vstack(annotation_contour.to_numpy())
        try :
            scribble_tumor, annotation, _, _ = s.scribble(contour_tissue)
        except: 
            scribble_tumor = None
        if scribble_tumor is not (None):
            scribble_tumor = scribble_tumor
            scribble_tumor = np.expand_dims(scribble_tumor, axis=1)
            annotation = np.expand_dims(annotation, axis=1)
            annotations_tumor.append(annotation)
            scribbles_tumor.append(scribble_tumor)

    ### Generate a scribble in a healthy region
    try:
        scribble_healthy, _, _, _ = s.scribble(annotation_healthy.squeeze())
    except: 
        scribble_healthy = None
    return (annotations_tumor, scribbles_tumor, annotation_healthy, scribble_healthy)
