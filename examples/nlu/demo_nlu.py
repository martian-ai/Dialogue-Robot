import argparse
import sys
sys.path.append("../..")

from solutions.nlu.traditional import traditional_to_simple
from solutions.nlu.segment import text_segment
from solutions.nlu.convert import string_upper
from solutions.nlu.lexical import seg, pos, ner

if __name__ == "__main__":
    print(seg('今天上班给老人让坐，四十分鐘的車程')) 
    print(pos('今天上班给老人让坐，四十分鐘的車程')) 
    print(ner('今天上班给老人让坐，四十分鐘的車程')) 