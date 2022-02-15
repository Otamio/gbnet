__all__ = ['extract_attributes', 'match_roi', 'inject_attribute',
           'attrnet_train', 'attribute_generate', 'graph_generate']

objects_zip_path = "resource/labeled/objects.v3.json.zip"
attributes_zip_path = "resource/labeled/attributes.json.zip"
metadata_zip_path = "resource/labeled/image_data.v2.json.zip"
graph_h5_raw_path = "resource/labeled/VG-SGG.h5"
vgg_json_raw_path = "resource/labeled/VG-SGG-dicts.json"
image_data_source_path = "resource/labeled/image_data.json"

extract_attribute_path = "temporary/attributes"
match_roi_result_path = "temporary/matching"
graph_h5_result_path = "temporary/processed/VG-SGG.h5"
vgg_json_processed_path = "temporary/processed/VG-SGG-dicts.json"
image_data_output_path = "temporary/processed/image_data.json"
attrnet_params = "temporary/output/attrnet.model"
