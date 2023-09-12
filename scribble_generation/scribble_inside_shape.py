import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import xml.etree.ElementTree as ET
import networkx as nx
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import triangulate
import random
from itertools import combinations
from scipy.interpolate import interp1d
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import *
from scipy.interpolate import CubicSpline

warnings.filterwarnings("ignore")


class Scribble:
    def __init__(
        self, filename, percent, split, show=False
    ):
        self.split = split
        self.filename = filename
        self.percent = percent
        self.show = show
        self.path_camelyon = path_camelyon

        if self.split == "train":
            self.path_annotations = path_annotations_train
        else:
            self.path_annotations = path_annotations_test

    def create_dataframe_annotations(self):
        path_annotation = os.path.join(self.path_annotations, self.filename + ".xml")
        tree = ET.parse(path_annotation)
        root = tree.getroot()
        list_annotations = []
        list_x = []
        list_y = []
        dataframe_tot = pd.DataFrame()
        for i, coordinates in enumerate(root.iter("Coordinates")):
            dataframe = pd.DataFrame(columns=["Annotation " + str(i)])
            for j, coord in enumerate(coordinates.iter("Coordinate")):
                attribs = coord.attrib
                x = float(attribs["X"])
                y = float(attribs["Y"])
                dataframe = dataframe.append(
                    {"Annotation " + str(i): [x, y]}, ignore_index=True
                )
            dataframe_tot = pd.concat([dataframe_tot, dataframe], axis=1)
        return dataframe_tot

    def point_on_triangle2(self, list_):
        """
        Random point on the triangle with vertices pt1, pt2 and pt3.
        """
        pt1 = list_[0]
        pt2 = list_[1]
        pt3 = list_[3]
        x, y = random.random(), random.random()
        q = abs(x - y)
        s, t, u = q, 0.5 * (x + y - q), 1 - 0.5 * (q + x + y)
        return (
            s * pt1[0] + t * pt2[0] + u * pt3[0],
            s * pt1[1] + t * pt2[1] + u * pt3[1],
        )

    def create_delaunay_inside(self, downsample):
        # Creating the polygon
        res_intersection = [Polygon(downsample)]
        res_intersection_gdf = gpd.GeoDataFrame()
        res_intersection_gdf.geometry = res_intersection
        # Create ID to identify overlapping polygons
        res_intersection_gdf["TRI_ID"] = res_intersection_gdf.index
        # List to keep triangulated geometries
        tri_geom = []
        # List to keep the original IDs
        tri_id = []
        # Triangulate single or multi-polygons
        for i, rows in res_intersection_gdf.iterrows():
            tri_ = triangulate(res_intersection_gdf.geometry.values[i])
            tri_geom.append(tri_)
            for id_length in range(0, len(tri_)):
                tri_id.append(res_intersection_gdf.TRI_ID.values[i])
        # Check if it is a single or multi-polygon
        len_list = len(tri_geom)
        tri_geom = np.array(tri_geom).flatten().tolist()
        # unlist geometries for multi-polygons
        if len_list > 1:
            tri_geom = [item for sublist in tri_geom for item in sublist]
        # Create triangulated polygons
        polygon = gpd.GeoDataFrame(tri_geom)
        polygon = polygon.set_geometry(tri_geom)
        if polygon.shape[1] == 2:
            del polygon[0]
            # Assign original IDs to each triangle
            polygon["TRI_ID"] = tri_id
            # Create new ID for each triangle
            polygon["LINK_ID"] = polygon.index
            # Create centroids from all triangulated polygons
            polygon["centroid"] = polygon.centroid
            polygon_centroid = polygon.set_geometry("centroid")
            del polygon_centroid["geometry"]
            del polygon["centroid"]
            polygon
            # Find triangle centroids inside original polygon
            polygon_join = gpd.sjoin(
                polygon_centroid[["centroid", "TRI_ID", "LINK_ID"]],
                res_intersection_gdf[["geometry", "TRI_ID"]],
                how="inner",
                op="within",
            )

            # Remove overlapping from other polygons (Necessary for multi-polygons overlapping or close to each other)
            polygon_join = polygon_join[
                polygon_join["TRI_ID_left"] == polygon_join["TRI_ID_right"]
            ]
            # Remove overload triangles from same polygons
            polygon = polygon[polygon["LINK_ID"].isin(polygon_join["LINK_ID"])]
            polygon = polygon.reset_index(drop=True)
            if self.show == True:
                fig, axs = plt.subplots()
                axs.set_aspect("equal", "datalim")
                for u in range(polygon.shape[0]):
                    x, y = polygon.loc[u]["geometry"].exterior.xy
                    axs.fill(x, y, alpha=0.5, color=np.random.randint(0, 255, 3) / 255)
                plt.axis('off')
                plt.show()
            return polygon
        else:
            return None

    def create_polygon_df_graph_df(self, polygon):
        polygon["Neighbors"] = False
        for u in range(polygon.shape[0]):
            geometry = polygon["geometry"].values[u]
            polygon["isNeighbor"] = polygon.apply(
                lambda row: len(
                    set(geometry.boundary.coords).intersection(
                        row["geometry"].boundary.coords
                    )
                )
                >= 2,
                axis=1,
            )
            polygon["Neighbors"][u] = np.array(
                polygon[polygon["isNeighbor"] == True].index
            )
            polygon["random_point"] = polygon.apply(
                lambda row: self.point_on_triangle2(
                    list(row["geometry"].boundary.coords)
                ),
                axis=1,
            )

        polygon["nb_neighbor"] = polygon.apply(
            lambda row: len(row["Neighbors"]), axis=1
        )
        list_isolated_edges = list(polygon[polygon["nb_neighbor"] == 2].index)
        graph_df = pd.DataFrame()
        graph_df["node"] = polygon.index
        graph_df["neighbors"] = polygon["Neighbors"]
        graph_df["points"] = polygon["random_point"]
        graph_df = graph_df.explode("neighbors")
        graph_df = graph_df[graph_df.node != graph_df.neighbors]
        graph_df["edge"] = graph_df[["node", "neighbors"]].values.tolist()

        return graph_df, polygon, list_isolated_edges

    def find_longest_path(self, graph_df, list_isolated_edges, net):
        dictionnary = graph_df["points"].drop_duplicates().to_dict()
        res = list(combinations(list_isolated_edges, 2))
        df = pd.DataFrame()
        df["pairs"] = res
        list_shortest_pathes = []
        for pair in list(df["pairs"]):
            source, target = pair
            try:
                sp = nx.shortest_path_length(net, source, target)
                list_shortest_pathes.append(sp)
            except:
                list_shortest_pathes.append(0)
        df["shortest_path"] = list_shortest_pathes
        source, target = tuple(
            df[df["shortest_path"] == df["shortest_path"].max()]["pairs"]
        )[0]
        path = nx.shortest_path(net, source, target)
        return path, dictionnary

    def draw_longest_path(self, path, net, dictionnary):
        path_edges = list(zip(path, path[1:]))
        net = net.to_undirected()
        plot = nx.draw_networkx_edges(net, pos=dictionnary, edgelist=path_edges)
        plt.axis("equal")
        plt.show()

    def interpolation_points_scribble(self, coordinates, contour, nb_):
        contour = np.vstack(np.array(contour))
        # Linear length along the line:
        distance = np.cumsum(np.sqrt(np.sum(np.diff(coordinates, axis=0) ** 2, axis=1)))
        distance = np.insert(distance, 0, 0) / distance[-1]
        # Interpolation for different methods:
        alpha = np.linspace(0, 1, nb_)
        interpolator = CubicSpline(distance, coordinates) 
        interpolated_points = interpolator(alpha)
        return interpolated_points, contour

    # def scribble_tumor(self, annotation, ps=512, ov=0.8, nb_=10000):

    #     arr = np.vstack(annotation.to_numpy())
    #     length = arr.shape[0]
    #     ni = 20

    #     if length < ni:
    #         downsample = arr

    #     else:
    #         ind__ = np.linspace(0, length - 1, ni).astype(int)
    #         downsample = np.array(arr[ind__])

    #     try:
    #         polygon = self.create_delaunay_inside(downsample)
    #         graph_df, polygon, list_isolated_edges = self.create_polygon_df_graph_df(
    #             polygon
    #         )

    #         net = nx.from_pandas_edgelist(graph_df, source="node", target="neighbors")
    #         net = net.to_directed()

    #         path, dictionnary = self.find_longest_path(
    #             graph_df, list_isolated_edges, net
    #         )
    #         coordinate_df = graph_df[graph_df["node"].isin(path)][
    #             ["node", "points"]
    #         ].drop_duplicates()
    #         list_coordinates = []

    #         for node in path:
    #             list_coordinates.append(
    #                 coordinate_df[coordinate_df["node"] == node]["points"].tolist()
    #             )
    #         coordinates = np.array(list_coordinates).squeeze()

    #         ret, shape = self.interpolation_points_scribble(coordinates, annotation, nb_=nb_)

    #         arr = ret[self.interpolation_method]
    #         length = np.sum(np.sqrt(np.sum((arr[:-1] - arr[1:]) ** 2, axis=1)))

    #         nb_patches = int(length / (ps * (1 - ov))) + 1
    #         indices = np.linspace(1, nb_, nb_patches).astype(int) - 1
    #         arr = np.array([arr[i] for i in indices])
    #         ret[self.interpolation_method] = arr
    #         return ret, shape

    #     except:
    #         return None, None

    def scribble(self, annotation, ps=ps, ov=0.8):
        
        """
        Input: the annotation of the healthy or tumor region

        Ouput: the contour of the annotation
               the scribble inside the contour

        """

        #### We only select ni = 20 points on the contour of the annotation
        arr = annotation
        length = arr.shape[0]
        ni = 15

        if length < ni:
            downsample = arr

        else:
            top = arr[:ni]
            bot = arr[-ni:]

            new_arr = np.concatenate([bot, arr, top])
            distance = np.cumsum(np.sqrt(np.sum(np.diff(new_arr, axis=0) ** 2, axis=1)))
            distance = np.insert(distance, 0, 0) / distance[-1]
            alpha = np.linspace(distance[ni], distance[-ni], ni)
            downsample = interp1d(distance, new_arr, kind="linear", axis=0)(alpha)
        #### Creation of the scribble

        polygon = self.create_delaunay_inside(downsample)
        graph_df, polygon, list_isolated_edges = self.create_polygon_df_graph_df(
            polygon
        )
        net = nx.from_pandas_edgelist(graph_df, source="node", target="neighbors")
        net = net.to_directed()
        path, dictionnary = self.find_longest_path(graph_df, list_isolated_edges, net)
        coordinate_df = graph_df[graph_df["node"].isin(path)][
            ["node", "points"]
        ].drop_duplicates()
        list_coordinates = []
        for node in path:
            list_coordinates.append(
                coordinate_df[coordinate_df["node"] == node]["points"].tolist()
            )
        coordinates = np.array(list_coordinates).squeeze()

        ### coordinates are the coordinate of the points randomly sampled inside each triangle
        ### We interpolate the extracted coordinates and get rid of the top and bottom part of the scribbles to get the final scribble:

        interpolated_points, contour = self.interpolation_points_scribble(
            coordinates, downsample, nb_=10000
        )
        arr = interpolated_points
        nb = arr.shape[0]
        remove = int((nb * self.percent)/2)
        scribble = arr[remove : nb - remove, :]
        length = np.sum(np.sqrt(np.sum((arr[:-1] - arr[1:]) ** 2, axis=1)))
        nb_patches = int(length / (ps * (1 - ov))) + 1
        indices = np.linspace(1, scribble.shape[0], nb_patches).astype(int) - 1
        arr = scribble[indices]
        return arr, contour, scribble, coordinates
