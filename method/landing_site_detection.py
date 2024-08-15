import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from method.utils.utils import  process_sg, process_so
import os

class ImageProcessor:
    def __init__(self, class_scores, class_mapping, image_path, tile_size_sg, step_size_sg, tile_size_of, region_size_of):
        self.class_scores = class_scores
        self.class_mapping = class_mapping
        self.image_path = image_path
        self.tile_size_sg = tile_size_sg
        self.step_size_sg = step_size_sg
        self.tile_size_of = tile_size_of
        self.region_size_of = region_size_of

        if isinstance(self.image_path, np.ndarray):
            self.image = Image.fromarray(self.image_path)
        else:
            self.image = Image.open(self.image_path)

        # Get the size of the image
        self.image_size = self.image.size[::-1]

    def calculate_scale(self, so, sc):
        row_scale = sc.shape[0] / so.shape[0]
        col_scale = sc.shape[1] / so.shape[1]
        return row_scale, col_scale

    def resize_array(self, so, row_scale, col_scale):
        so_resized = zoom(so, (row_scale, col_scale))
        return so_resized

    def normalize_array(self, array):
        array_scaled = (array - np.min(array)) / (np.max(array) - np.min(array)) * (28 - (-28)) + (-28)
        return array_scaled

    def calculate_s(self, sc_scaled, so_resized_scaled):
        s = 0.5 * (sc_scaled + so_resized_scaled)
        return s

    def find_safe_tiles(self, s):
        safe_tiles = np.argwhere(s >= 14.4)
        return safe_tiles

    def find_top_tiles(self, s):
        top_tiles = np.unravel_index(np.argpartition(s.flatten(), -3)[-3:], s.shape)
        return top_tiles
    
    def process_sg(self, np_segmentation_result):
        return process_sg(np_segmentation_result, self.class_scores, self.class_mapping, self.image_size, self.tile_size_sg, self.step_size_sg, top_n=3)

    def process_so(self, flow):
        return process_so(flow,  self.tile_size_of, self.region_size_of)


    def process_images(self, flow_result, segmentation_result):
        if isinstance(segmentation_result, np.ndarray):
            np_segmentation_result = segmentation_result
        elif isinstance(segmentation_result, (str, bytes, os.PathLike)):
            np_segmentation_result = np.load(segmentation_result)
        else:
            raise TypeError(f"Expected segmentation_result to be a path to a .npy file or a numpy array, got {type(segmentation_result)} instead.")
        
        if isinstance(flow_result, np.ndarray):
            flow_result_array = flow_result
        elif isinstance(flow_result, (str, bytes, os.PathLike)):
            flow_result_array = np.load(flow_result)
        else:
            raise TypeError(f"Expected flow_result to be a path to a .npy file or a numpy array, got {type(flow_result)} instead.")
        
        sg = process_sg(np_segmentation_result, self.class_scores, self.class_mapping, self.image_size, self.tile_size_sg, self.step_size_sg, top_n=3)
        so = process_so(flow_result_array, self.tile_size_of, self.region_size_of)
        row_scale, col_scale = self.calculate_scale(so, sg)
        so_resized = self.resize_array(so, row_scale, col_scale)
        sg_scaled = self.normalize_array(sg)
        so_resized_scaled = self.normalize_array(so_resized)
        s = self.calculate_s(sg_scaled, so_resized_scaled)
        safe_tiles = self.find_safe_tiles(s)
        top_tiles = self.find_top_tiles(s)
        return safe_tiles, top_tiles, sg_scaled, s, so_resized_scaled 