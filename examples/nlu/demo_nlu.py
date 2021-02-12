import argparse
import sys
sys.path.append("../..")

from solutions.nlu.traditional import traditional_to_simple
from solutions.nlu.segment import text_segment
from solutions.nlu.convert import string_upper
from solutions.nlu.lexical import segment, pos_tagging, name_entity, lexical_analysis
from solutions.nlu.syntax import dp


if __name__ == "__main__":
    print(segment('今天上班给老人让坐，四十分鐘的車程')) 
    print(pos_tagging('今天上班给老人让坐，四十分鐘的車程')) 
    print(name_entity('今天上班给老人让坐，四十分鐘的車程')) 
    print(lexical_analysis('今天上班给老人让坐，四十分鐘的車程')) 
    print(dp('今天上班给老人让坐，四十分鐘的車程', show=True))