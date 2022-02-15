import h5py
import pipeline
import json
import numpy as np
from tqdm import tqdm


def inject_attributes(use_cache=True, debug=False):

    # Get object ids from match roi
    with open(f'{pipeline.match_roi_result_path}/matched_ids.json') as fd:
        object_ids = json.load(fd)
    # Get object -> attribute pairs
    with open(f'{pipeline.extract_attribute_path}/object_id_with_attributes.json') as fd:
        object_id_2_attributes = json.load(fd)
        object_id_2_attributes = {int(k): v for k, v in object_id_2_attributes.items()}
    # Get attributes selected
    with open(f'{pipeline.extract_attribute_path}/select_attributes.json') as fd:
        attributes_selected = json.load(fd)

    # Generate Attribute Table
    attribute_table = {attr: i+1 for i, attr in enumerate(attributes_selected)}
    attribute_table["__blank__"] = 0

    # Generate Metadata
    with open(pipeline.vgg_json_raw_path) as fd:
        vgg_metadata = json.load(fd)
    vgg_metadata['attribute_to_idx'] = attribute_table
    vgg_metadata['idx_to_attribute'] = {v: k for k, v in attribute_table.items()}
    with open(pipeline.vgg_json_processed_path, 'w') as fd:
        json.dump(vgg_metadata, fd, indent=2)

    # Move data
    with open(pipeline.image_data_source_path) as fd:
        with open(pipeline.image_data_output_path, 'w') as out:
            json.dump(json.load(fd), out, indent=2)

    # Generate necessary data for us to estimate
    attributes = []
    attributes_id_start = []
    attributes_id_end = []

    # Generate attributes
    for object_id in tqdm(object_ids):
        # If object_id is not found, or it does not contain any data, skip
        if object_id == -1 or object_id not in object_id_2_attributes:
            attributes_id_start.append(-1)
            attributes_id_end.append(-1)
        else:
            start_id = len(attributes)
            end_id = len(attributes) - 1
            for attr in object_id_2_attributes[object_id]:
                if attr in attribute_table:
                    attributes.append(attribute_table[attr])
                    end_id += 1
            if start_id <= end_id:
                attributes_id_start.append(start_id)
                attributes_id_end.append(end_id)
            else:
                attributes_id_start.append(-1)
                attributes_id_end.append(-1)

    # Load h5copy
    with h5py.File(pipeline.graph_h5_raw_path, 'r') as roi_h5:
        with h5py.File(pipeline.graph_h5_result_path, 'w') as roi:
            for key in roi_h5:
                roi.create_dataset(key, data=np.array(roi_h5[key]))
            roi.create_dataset('box_to_object_id', data=np.array(object_ids))
            roi.create_dataset('attributes', data=np.array(attributes))
            roi.create_dataset('box_to_first_attribute', data=np.array(attributes_id_start))
            roi.create_dataset('box_to_last_attribute', data=np.array(attributes_id_end))
