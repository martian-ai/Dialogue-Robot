import argparse
import sys
sys.path.append("../..")

from solutions.nlu.traditional import traditional_to_simple
from solutions.nlu.segment import text_segment
from solutions.nlu.convert import string_upper
from solutions.nlu.


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='arguments for nlu')
    parser.add_argument('--input', type=str, help='input file name')
    parser.add_argument('--output', type=str, help='output file name')
    args = parser.parse_args()

    words = []
    with open(args.input, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line is "":
                continue
            line = traditional_to_simple(line)
            line = string_upper(line)
            cut_line = text_segment(line)
            words.extend([  item for item in cut_line if len(item) < 7])

    words = list(set(words))
    words = sorted(words)

    with open(args.output, mode='w', encoding='utf-8') as f:
        for item in words:
            f.write(item + '\n')


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(description='arguments for nlu')
#     parser.add_argument('--input', type=str, help='input file name')
#     parser.add_argument('--output', type=str, help='output file name')
#     args = parser.parse_args()

#     words = []
#     with open(args.input, mode='r', encoding='utf-8') as f:
#         lines = f.readlines()
#         for line in lines:
#             line = line.strip()
#             if line is "":
#                 continue
#             line = traditional_to_simple(line)
#             line = string_upper(line)
#             cut_line = text_segment(line)
#             words.extend([  item for item in cut_line if len(item) < 7])

#     words = list(set(words))
#     words = sorted(words)

#     with open(args.output, mode='w', encoding='utf-8') as f:
#         for item in words:
#             f.write(item + '\n')