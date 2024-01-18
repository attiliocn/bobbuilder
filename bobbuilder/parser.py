import argparse

parser = argparse.ArgumentParser()
parser.add_argument('core', help='core file')
parser.add_argument('--core-atoms', help='core atoms')
parser.add_argument('--replace-at', action='append', help='core atoms')

parser.add_argument('--fragment', action='append', help='fragment file')
parser.add_argument('--remove-h', action='append', choices=['yes', 'no'], help='remove terminal hydrogen')
parser.add_argument('--connect-atom', action='append', help='fragment atom')
parser.add_argument('--axis-atom', action='append', help='fragment axis')

args = parser.parse_args()

print(args)
