import os
import re
import subprocess
from pathlib import Path
from typing import FrozenSet


def set_class_attr_to_nominal(input_file_path: Path, output_file_path: Path, weka_jar_path: Path):
    """
    Applies Weka's NumericToNominal filter on the last (class) attribute of the given file.
    The input can be a CSV or an ARFF file, as Weka can handle both.
    The output file will be in the ARFF format.

    :param input_file_path: CSV or ARFF file
    :param output_file_path: ARFF output file with the applied filter
    :param weka_jar_path: Path to the Weka's jar
    """
    subprocess.call(['java', '-cp', weka_jar_path, 'weka.filters.unsupervised.attribute.NumericToNominal',
                     '-R', 'last', '-i', input_file_path, '-o', output_file_path])


def select_top_correlating_attrs(input_file_path: Path, output_file_path: Path, attributes_to_take: int,
                                 weka_jar_path: Path) -> FrozenSet[int]:
    """
    Applies Weka's attribute selection using CorrelationAttributeEval evaluator.

    :param input_file_path: CSV or ARFF file
    :param output_file_path: ARFF output file
    :param attributes_to_take: Number of attributes to take
    :param weka_jar_path: Path to the Weka's jar
    :return: Immutable set of attributes that were selected by the evaluator. Does not include the class attribute.
    """
    RANKER_THRESHOLD = '-1.7976931348623157E308'  # Picked by Weka automatically
    search_method = f'weka.attributeSelection.Ranker -T {RANKER_THRESHOLD} -N {attributes_to_take}'
    command = ' '.join(['java', '-cp', str(weka_jar_path), 'weka.filters.supervised.attribute.AttributeSelection',
                        '-E', 'weka.attributeSelection.CorrelationAttributeEval', '-S', f'"{search_method}"',
                        '-i', str(input_file_path), '-o', str(output_file_path)])
    os.system(command)
    return get_attributes_of_arff_file(output_file_path)


def get_attributes_of_arff_file(file_path: str) -> FrozenSet[int]:
    """
    Finds what attributes are used by the given file
    :param file_path: ARFF file
    :return: Immutable set of attributes without the class attribute.
    """
    if not file_path.endswith('.arff'):
        raise ValueError("Only arff files are supported by this function")

    attributes = set()
    regex = re.compile(r'(\d+) numeric')
    with open(file_path) as file:
        for line in file:
            if '@data' in line:
                break  # End of attribute information. We can finish reading the file here.
            if '@attribute' in line:
                match = regex.search(line)
                if match:
                    attributes.add(int(match.group(1)))
    return frozenset(attributes)  # Return as immutable


if __name__ == "__main__":
    # TODO: Remove after implementing all things
    weka_path = Path.home().joinpath('weka-3-8-4/weka.jar')
    input_path = Path.home().joinpath('PycharmProjects/F21DL-CW2/data/4000_instances/train.csv')
    filtered_output_path = Path.home().joinpath('PycharmProjects/F21DL-CW2/data/!filtered.arff')
    corr_output_path = Path.home().joinpath('PycharmProjects/F21DL-CW2/data/!corr.arff')

    set_class_attr_to_nominal(input_path, filtered_output_path, weka_path)
    attributes = select_top_correlating_attrs(filtered_output_path, corr_output_path, 50, weka_path)
    print(attributes)
    print(len(attributes))
