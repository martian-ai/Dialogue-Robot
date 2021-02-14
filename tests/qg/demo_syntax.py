import argparse
import sys
sys.path.append("../..")

from solutions.qg.syntax import question_generate_by_syntax

if __name__ == "__main__":
    print(question_generate_by_syntax('国务院规定农名工的工资应当按时发放')) 