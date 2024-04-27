import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='My program',
                    description='It parses some arguments',
                    epilog='Text at the bottom of help')
    parser.add_argument("config", default="configs/default.yaml")
    parser.parse_args()
    print(parser.parse_args().config)