import os
import re
import subprocess
from pathlib import Path
from typing import FrozenSet


class Weka:
    def __init__(self, weka_jar_path: Path):
        self.weka_jar_path = weka_jar_path

    def set_class_attr_to_nominal(self, input_file_path: Path, output_file_path: Path):
        """
        Applies Weka's NumericToNominal filter on the last (class) attribute of the given file.
        The input can be a CSV or an ARFF file, as Weka can handle both.
        The output file will be in the ARFF format.

        :param input_file_path: CSV or ARFF file
        :param output_file_path: ARFF output file with the applied filter
        """
        subprocess.call(['java', '-cp', self.weka_jar_path, 'weka.filters.unsupervised.attribute.NumericToNominal',
                         '-R', 'last', '-i', input_file_path, '-o', output_file_path])

    def select_top_correlating_attrs(self, input_file_path: Path,
                                     output_file_path: Path, attributes_to_take: int) -> FrozenSet[int]:
        """
        Applies Weka's attribute selection using CorrelationAttributeEval evaluator.

        :param input_file_path: CSV or ARFF file
        :param output_file_path: ARFF output file
        :param attributes_to_take: Number of attributes to take
        :return: Immutable set of attributes that were selected by the evaluator. Does not include the class attribute.
        """
        RANKER_THRESHOLD = '-1.7976931348623157E308'  # Picked by Weka automatically
        search_method = f'weka.attributeSelection.Ranker -T {RANKER_THRESHOLD} -N {attributes_to_take}'
        command = ' '.join(
            ['java', '-cp', str(self.weka_jar_path), 'weka.filters.supervised.attribute.AttributeSelection',
             '-E', 'weka.attributeSelection.CorrelationAttributeEval', '-S', f'"{search_method}"',
             '-i', str(input_file_path), '-o', str(output_file_path)])
        os.system(command)
        return self.get_attributes_of_arff_file(output_file_path)

    @staticmethod
    def get_attributes_of_arff_file(file_path: Path) -> FrozenSet[int]:
        """
        Finds what attributes are used by the given file
        :param file_path: ARFF file
        :return: Immutable set of attributes without the class attribute.
        """
        if not file_path.name.endswith('.arff'):
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

    def filter_attributes(self, input_file_path: Path, output_file_path: Path, attributes: FrozenSet[int]):
        """
        Filters from the input file only the attributes that are in the given set and saves them to the output file.

        :param input_file_path: Input file to be processed. Can be either ARFF or CSV.
        :param output_file_path: ARFF Output file with only the selected attributes
        :param attributes: Attributes to take
        """
        # The -V flag inverts the selection.
        subprocess.call(['java', '-cp', str(self.weka_jar_path), 'weka.filters.unsupervised.attribute.Remove', '-V',
                         '-R', ','.join(str(attr + 1) for attr in attributes) + ',last', '-i', input_file_path, '-o',
                         output_file_path])
