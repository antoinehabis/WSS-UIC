import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import logging
from argparse import ArgumentParser
from shapely.geometry import Point, Polygon, MultiPoint, LineString
from cytomine import Cytomine
from cytomine.models import Annotation, AnnotationTerm, AnnotationCollection
from scipy.interpolate import interp1d
from config import *
from openslide import OpenSlide
import numpy as np

logging.basicConfig()
logger = logging.getLogger("cytomine.client")
logger.setLevel(logging.INFO)


class Send:
    def __init__(self, filename, split="train"):
        self.split = split
        self.filename = filename

        if self.split == "train":
            self.path_image = path_slide_tumor_train
        else:
            self.path_image = path_slide_tumor_test

        img = OpenSlide(os.path.join(self.path_image, self.filename,'.tif'))
        self.dim = img.dimensions[1]

    def delete_loops(self, shape):
        n = shape.shape[0]
        l = []
        for i in range(2, n - 1):
            C, D = shape[i], shape[i + 1]

            for j in range(i - 2):
                A, B = shape[j], shape[j + 1]
                line = LineString([A, B])
                other = LineString([C, D])
                inter = line.intersects(other)
                point = line.intersection(other)
                if inter:
                    l = l + list(np.arange(j + 1, i + 1))

        arr = list(np.arange(shape.shape[0]))
        indices = set(arr) - set(l)

        indices = list(indices)
        shape = shape[indices]
        distance = np.cumsum(np.sqrt(np.sum(np.diff(shape, axis=0) ** 2, axis=1)))
        distance = np.insert(distance, 0, 0) / distance[-1]

        # Interpolation for different methods:
        alpha = np.linspace(0, 1, n)
        shape = interp1d(distance, shape, kind="linear", axis=0)(alpha)

        return shape

    def preprocess_cytomine(
        self, shapes, scribbles, contour_foreground, scribble_foreground
    ):
        shapes = [shape.squeeze() for shape in shapes]
        for shape in shapes:
            shape[:, 1] = (self.dim - shape[:, 1]).astype(int)
        scribbles = [scribble.squeeze() for scribble in scribbles]
        for scribble in scribbles:
            scribble[:, 1] = self.dim - scribble[:, 1]
        contour_foreground = contour_foreground.squeeze()
        contour_foreground[:, 1] = self.dim - contour_foreground[:, 1]

        scribble_foreground = scribble_foreground.squeeze().astype(np.int)
        scribble_foreground[:, 1] = self.dim - scribble_foreground[:, 1]
        return shapes, scribbles, contour_foreground, scribble_foreground

    def fill_wkts(self, shapes, scribbles, contour_foreground, scribble_foreground):
        wkts = []

        for scribble in scribbles:
            poly = MultiPoint(scribble).wkt
            wkts.append(poly)

        for shape in shapes:
            poly = Polygon(self.delete_loops(shape)).wkt
            wkts.append(poly)

        poly1 = Polygon(self.delete_loops(contour_foreground.squeeze())).wkt
        poly2 = MultiPoint(scribble_foreground.squeeze()).wkt

        wkts.append(poly1)
        wkts.append(poly2)
        return wkts

    def send_annotation(self, list_annotations, id_image, id_project):
        pb_key = os.environ['PB_KEY']
        pv_key = os.environ['PV_KEY']
        host = "https://nsclc.cytomine.com/"

        with Cytomine(host=host, public_key=pb_key, private_key=pv_key) as cytomine:
            # We can also add multiple annotation in one request:
            annotations = AnnotationCollection()

            for annotation in list_annotations:
                annotations.append(
                    Annotation(
                        location=annotation, id_image=id_image, id_project=id_project
                    )
                )
                annotations.save()
        return "Annotations sent! You are the best Antoine"

    def send_annotations_to_cytomine(
        self,
        shapes,
        scribbles,
        contour_foreground,
        scribble_foreground,
        id_image,
        id_project,
    ):
        (
            shapes,
            scribbles,
            contour_foreground,
            scribble_foreground,
        ) = self.preprocess_cytomine(
            shapes, scribbles, contour_foreground, scribble_foreground
        )
        WKT = self.fill_wkts(shapes, scribbles, contour_foreground, scribble_foreground)

        self.send_annotation(WKT, id_image, id_project)
