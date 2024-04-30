# import code.data as data
import argparse
from training import train, test


if __name__ == '__main__':
    # TODO: parse arguments
    ap = argparse.ArgumentParser("Progressive Transformers")

    # Choose between Train and Test
    ap.add_argument("mode", choices=["train", "test"],
                    help="train a model or test")
    # Path to Config
    ap.add_argument("config_path", type=str,
                    help="path to YAML config file")

    # Optional path to checkpoint
    ap.add_argument("--ckpt", type=str,
                    help="path to model checkpoint")

    # TODO: process data
    args = ap.parse_args()

    
    # TODO: train model if in training mode, test model otherwise

    if args.mode == "train":
        train(cfg_file=args.config_path, ckpt=args.ckpt)
    # If Test
    elif args.mode == "test":
        test(cfg_file=args.config_path, ckpt=args.ckpt)
    else:
        raise ValueError("Unknown mode")

    # pass
