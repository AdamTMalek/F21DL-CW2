import subprocess
from pathlib import Path


def set_class_attr_to_nominal(input_file_path: Path, output_file_path: Path, weka_jar_path: Path):
    """
    Applies Weka's NumericToNominal filter on the last (class) attribute of the given file.
    The input can be a CSV or an ARFF file, as Weka can handle both.
    The output file will be in the ARFF format.

    :param input_file_path: CSV or ARFF file
    :param output_file_path: ARFF output file with the applied filter
    :param weka_jar_path: Path to the weka's jar
    """
    subprocess.call(['java', '-cp', weka_jar_path, 'weka.filters.unsupervised.attribute.NumericToNominal',
                     '-R', 'last', '-i', input_file_path, '-o', output_file_path])
