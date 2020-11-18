import os
import subprocess
from pathlib import Path


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
                                 weka_jar_path: Path):
    """
    Applies Weka's attribute selection using CorrelationAttributeEval evaluator.

    :param input_file_path: CSV or ARFF file
    :param output_file_path: ARFF output file
    :param attributes_to_take: Number of attributes to take
    :param weka_jar_path: Path to the Weka's jar
    :return:
    """
    RANKER_THRESHOLD = '-1.7976931348623157E308'  # Picked by Weka automatically
    search_method = f'weka.attributeSelection.Ranker -T {RANKER_THRESHOLD} -N {attributes_to_take}'
    command = ' '.join(['java', '-cp', str(weka_jar_path), 'weka.filters.supervised.attribute.AttributeSelection',
                        '-E', 'weka.attributeSelection.CorrelationAttributeEval', '-S', f'"{search_method}"',
                        '-i', str(input_file_path), '-o', str(output_file_path)])
    os.system(command)
