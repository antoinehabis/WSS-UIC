import argparse

parser = argparse.ArgumentParser(description='A test program.')

parser.add_argument("-s", "--split", help="Prints the supplied argument.", type=str)

args = parser.parse_args()

print(args.split)