__all__ = ['extract_attributes', 'match_roi']

objects_zip_path = "resource/labeled/objects.v3.json.zip"
attributes_zip_path = "resource/labeled/attributes.json.zip"
metadata_zip_path = "resource/labeled/image_data.v2.json.zip"
graph_h5_raw_path = "resource/labeled/VG-SGG.h5"
vgg_json_raw_path = "resource/labeled/VG-SGG-dicts.json"

extract_attribute_path = "temporary/attributes"
match_roi_result_path = "temporary/matching"
graph_h5_result_path = "resource/labeled/VG-SGG-attr.h5"
