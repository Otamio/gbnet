import h5py
import json
import pipeline
import zipfile
from tqdm import tqdm


def match_roi(debug=False):

    # Load object metadata
    with zipfile.ZipFile(pipeline.objects_zip_path) as z:
        filename = z.namelist()[0]
        with z.open(filename) as fd:
            objects_v3 = json.load(fd)

    with zipfile.ZipFile(pipeline.metadata_zip_path) as z:
        filename = z.namelist()[0]
        with z.open(filename) as fd:
            metadata_v2 = json.load(fd)

    with open(pipeline.vgg_json_raw_path, 'r') as vgg:
        vgg_metadata = json.load(vgg)

    # Inject objects with dimension data
    print('Calculating object sizes')
    for image_ind, image in tqdm(enumerate(objects_v3)):
        for obj in image['objects']:
            obj_w_1024 = obj['w'] / metadata_v2[image_ind]['width'] * 1024
            obj_h_1024 = obj['h'] / metadata_v2[image_ind]['width'] * 1024
            obj_xc_1024 = obj['x'] / metadata_v2[image_ind]['width'] * 1024 + obj_w_1024 / 2
            obj_yc_1024 = obj['y'] / metadata_v2[image_ind]['width'] * 1024 + obj_h_1024 / 2
            obj['boxes_1024'] = [obj_xc_1024, obj_yc_1024, obj_w_1024, obj_h_1024]

    corrupted_id = [1592, 1722, 4616, 4617]  # These are the ids that we should skip

    object_ids = []  # Data holder to store all matched object ids
    object_matched_pairs = []
    objects_v3_id, non_matches, matches = 0, 0, 0
    skips, img_matched = 0, 0
    with h5py.File(pipeline.graph_h5_raw_path, 'r') as roi_h5:
        for iii, (b, e) in tqdm(enumerate(zip(roi_h5['img_to_first_box'],
                                              roi_h5['img_to_last_box']))):

            # skip images that do not have any bounding boxes
            if b == -1 or e == -1:
                skips += 1
                objects_v3_id += 1
                continue
            else:
                img_matched += 1

            # skip corrupted images
            while objects_v3[objects_v3_id]['image_id'] in corrupted_id:
                objects_v3_id += 1

            object_list = objects_v3[objects_v3_id]['objects']
            k = 0  # Do not check frames after k
            for i in range(b, e+1):  # Check through all the ROIs
                flag_object_matched = False
                for j, obj in enumerate(object_list[k:]):
                    if abs(roi_h5['boxes_1024'][i] - obj['boxes_1024']).sum() < 10:
                        object_ids.append(obj['object_id'])
                        object_matched_pairs.append((vgg_metadata['idx_to_label'][str(roi_h5['labels'][i].item())],
                                                     obj['names'][0] if len(obj['names']) > 0 else None))
                        k += j + 1
                        flag_object_matched = True
                        break
                if flag_object_matched:
                    matches += 1
                else:
                    non_matches += 1
                    object_ids.append(-1)
                    object_matched_pairs.append((vgg_metadata['idx_to_label'][str(roi_h5['labels'][i].item())], None))

            objects_v3_id += 1

            if debug and (iii + 1) % 20000 == 0:
                print('Epoch:', iii, 'Matching %:', matches / (matches + non_matches))
                with open(f'{pipeline.match_roi_result_path}/matched_ids.json', 'w') as fd:
                    json.dump(object_ids, fd, indent=2)
                with open(f'{pipeline.match_roi_result_path}/matched_pairs.json', 'w') as fd:
                    json.dump(object_matched_pairs, fd, indent=2)

    print(objects_v3_id)

    print('###############################################')
    print('Report Match Result:')
    print('Image Matches:', img_matched)
    print('Image Skipped:', skips)
    print('Object Matches:', matches)
    print('Object Non-matches:', non_matches)
    print('Object Match Rate:', matches / (matches + non_matches))
    print('################################################')

    if debug:
        with open(f'{pipeline.match_roi_result_path}/matched_ids.json', 'w') as fd:
            json.dump(object_ids, fd, indent=2)
        with open(f'{pipeline.match_roi_result_path}/matched_pairs.json', 'w') as fd:
            json.dump(object_matched_pairs, fd, indent=2)
