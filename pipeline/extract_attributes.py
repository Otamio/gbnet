import json
import zipfile
import pipeline
from collections import defaultdict
from tqdm import tqdm


def extract_top_attributes(k=50, debug=False):

    # Load attribute metadata
    with zipfile.ZipFile(pipeline.attributes_zip_path) as z:
        filename = z.namelist()[0]
        with z.open(filename) as fd:
            attributes_v2 = json.load(fd)

    # Counting
    count_attributes = defaultdict(int)
    count_objects = defaultdict(int)
    count_obj_attribute_pairs = defaultdict(lambda: defaultdict(int))
    object_id_2_attributes = defaultdict(list)

    # Load data and their attributes
    print('Scanning through the images......')
    for image in tqdm(attributes_v2):
        for roi in image['attributes']:
            if 'attributes' in roi:  # Get metadata from objects with labels
                for attr in roi['attributes']:
                    # Increment object
                    obj = roi['names'][0].strip()
                    count_objects[obj] += 1
                    # Increment attribute
                    attr = attr.strip().lower()
                    count_attributes[attr] += 1
                    # Increment object to attribute pairs
                    count_obj_attribute_pairs[obj][attr] += 1
                    # Record the attributes for each object Id
                    object_id_2_attributes[roi['object_id']].append(attr)

    top_attributes = sorted([x[0] for x in sorted(count_attributes.items(), key=lambda x: x[1], reverse=True)[:k]])

    print('Attribute extraction... Done...')

    if debug:
        with open(f'{pipeline.extract_attribute_path}/attribute_counts.json', 'w') as fd:
            json.dump(count_attributes, fd, indent=2)
        with open(f'{pipeline.extract_attribute_path}/object_counts.json', 'w') as fd:
            json.dump(count_objects, fd, indent=2)
        with open(f'{pipeline.extract_attribute_path}/object_attribute_pairs.json', 'w') as fd:
            json.dump(count_obj_attribute_pairs, fd, indent=2)
        with open(f'{pipeline.extract_attribute_path}/object_id_with_attributes.json', 'w') as fd:
            json.dump(object_id_2_attributes, fd, indent=2)
        with open(f'{pipeline.extract_attribute_path}/select_attributes.json', 'w') as fd:
            json.dump(top_attributes, fd, indent=2)
