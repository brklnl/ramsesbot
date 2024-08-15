import numpy as np

from skimage.util import view_as_windows
import tqdm




def get_class_probabilities(tile, class_scores):
    # Calculate the histogram of class occurrences within the tile
    class_histogram = np.bincount(tile.ravel(), minlength=len(class_scores))
    # Normalize the histogram to get class probabilities
    class_probabilities = class_histogram / np.sum(class_histogram)
    # Check the number of pixels of class 3
    # if class_histogram[3] > 2000:
    #     class_scores[3] = -1
    return class_probabilities

def get_surrounding_regions(tiles, i, j):
    surrounding_regions = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if (di, dj) != (0, 0) and 0 <= i + di < tiles.shape[0] and 0 <= j + dj < tiles.shape[1]:
                surrounding_regions.append(tiles[i + di, j + dj])
    return surrounding_regions

def process_sg(np_segmentation_result, class_scores, class_mapping, image_size, tile_size, step_size, top_n=3):
    segmentation_result_cropped = np_segmentation_result[:image_size[0], :image_size[1]]


    tiles = view_as_windows(segmentation_result_cropped, tile_size, step=step_size)

    # Initialize an empty array for the final scores
    final_scores = np.zeros((tiles.shape[0], tiles.shape[1]))

    # For each tile
    for i in range(tiles.shape[0]):
        for j in range(tiles.shape[1]):
            # Get the tile and its surrounding regions
            tile = tiles[i, j]
            surrounding_regions = get_surrounding_regions(tiles, i, j)  

            combined_probabilities = 4 * get_class_probabilities(tile, class_scores)
            for region in surrounding_regions[:4]:
                combined_probabilities += 2 * get_class_probabilities(region, class_scores)
            for region in surrounding_regions[4:]:
                combined_probabilities += get_class_probabilities(region, class_scores)

            # Calculate the temporal score
            temporal_score = np.sum(combined_probabilities * np.array(list(class_scores.values())))

            # Calculate the final score
            surrounding_scores = [np.sum(get_class_probabilities(region, class_scores) * np.array(list(class_scores.values()))) for region in surrounding_regions]
            final_score = temporal_score + 0.1 * np.sum(surrounding_scores)

            # Store the final score
            final_scores[i, j] = final_score

    return final_scores


import numpy as np
import tqdm

def calculate_so_temporal_scores(flow, tile_size, region_size):
    # Split the flow into two channels
    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]
    
    # Initialize the score arrays for both channels
    so_x = np.zeros((flow_x.shape[0] // tile_size, flow_x.shape[1] // tile_size))
    so_y = np.zeros((flow_y.shape[0] // tile_size, flow_y.shape[1] // tile_size))
    
    for i in tqdm.tqdm(range(0, flow_x.shape[0] - tile_size + 1, tile_size)):
        for j in range(0, flow_x.shape[1] - tile_size + 1, tile_size):
            regions_x = [flow_x[max(0, i - region_size // 2):min(flow_x.shape[0], i + region_size // 2),
                                max(0, j - region_size // 2):min(flow_x.shape[1], j + region_size // 2)] for _ in range(4)]
            regions_y = [flow_y[max(0, i - region_size // 2):min(flow_y.shape[0], i + region_size // 2),
                                max(0, j - region_size // 2):min(flow_y.shape[1], j + region_size // 2)] for _ in range(4)]
            
            epsilon = 1e-7  # small constant
            cvs_x = [np.nanstd(region_flow) / (np.nanmean(region_flow) + epsilon) for region_flow in regions_x]
            cvs_y = [np.nanstd(region_flow) / (np.nanmean(region_flow) + epsilon) for region_flow in regions_y]
            
            median_cv_x = np.median(cvs_x)
            median_cv_y = np.median(cvs_y)
            
            cvs_x = [cv if cv <= median_cv_x else 0.025 for cv in cvs_x]
            cvs_y = [cv if cv <= median_cv_y else 0.025 for cv in cvs_y]
            
            so_bar_x = np.sum(cvs_x)
            so_bar_y = np.sum(cvs_y)
            
            so_x[i // tile_size, j // tile_size] = 288 * (so_bar_x - 0.1)
            so_y[i // tile_size, j // tile_size] = 288 * (so_bar_y - 0.1)
    
    # Combine the scores from both channels
    so = (so_x + so_y) / 2
    return so

def process_so(flow, tile_size, region_size):
    so = calculate_so_temporal_scores(flow, tile_size, region_size)
    return so

def divide_image_into_tiles(image, tile_size):
    tiles = []
    h, w = image.shape[:2]
    for y in range(0, h, tile_size[0]):
        for x in range(0, w, tile_size[1]):
            tile = image[y:y + tile_size[0], x:x + tile_size[1]]
            tiles.append((x, y, tile))
    return tiles

def calculate_flow_magnitude(flow_image):
    flow_magnitude = np.sqrt(flow_image[..., 0]**2 + flow_image[..., 1]**2)
    return flow_magnitude

def calculate_mean_magnitude(tile):
    return np.mean(tile)

def find_safe_zones_mean_magnitude_v2(flow_image, tile_size, extension=30):
    flow_magnitude = calculate_flow_magnitude(flow_image)
    h, w = flow_magnitude.shape[:2]
    num_tiles_y = (h + tile_size[0] - 1) // tile_size[0]
    num_tiles_x = (w + tile_size[1] - 1) // tile_size[1]
    safety_scores = np.zeros((num_tiles_y, num_tiles_x))
    max_magnitude = np.max(flow_magnitude)
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            y_start = max(0, y * tile_size[0] - extension)
            y_end = min(h, (y + 1) * tile_size[0] + extension)
            x_start = max(0, x * tile_size[1] - extension)
            x_end = min(w, (x + 1) * tile_size[1] + extension)
            tile = flow_magnitude[y_start:y_end, x_start:x_end]
            mean_magnitude = calculate_mean_magnitude(tile)
            safety_scores[y, x] = max_magnitude - mean_magnitude
    return safety_scores

