from logging import getLogger

from fire import Fire

getLogger("fairseq").setLevel("ERROR")


def preprocess():
    from latopia.cli.subcommands import preprocess

    return preprocess.run()


def train():
    from latopia.cli.subcommands import train

    return train.run()


def infer():
    from latopia.cli.subcommands import infer

    return infer.run()


def webui():
    from latopia.cli.subcommands import webui

    return webui.run()


def main():
    return Fire(
        {"preprocess": preprocess, "train": train, "infer": infer, "webui": webui}
    )


if __name__ == "__main__":
    main()
