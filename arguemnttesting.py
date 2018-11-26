import argparse
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_filename", help="print this requires the input filename")
    parser.add_argument("config_file", help="this reuquires the excel configuration file")
    args = parser.parse_args()
    print args.input_filename
    print args.config_file
    return args.input_filename, args.config_file