import pipeline.extract_attributes
import pipeline.match_roi
import pipeline.inject_attribute
import pipeline.attrnet_train
import pipeline.attribute_generate
import sys


def main():
    if sys.argv[1] == "extract":
        pipeline.extract_attributes.extract_top_attributes(50, True)
    elif sys.argv[1] == "match":
        pipeline.match_roi.match_roi(debug=True)
    elif sys.argv[1] == "inject":
        pipeline.inject_attribute.inject_attributes()
    elif sys.argv[1] == "attrnet":
        pipeline.attrnet_train.main()
    elif sys.argv[1] == "generate":
        pipeline.attribute_generate.main()


if __name__ == "__main__":
    main()
