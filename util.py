import logging
import pathlib
import matplotlib.pyplot as plt

DB_PATH = pathlib.Path(__file__).parent.absolute()


def get_logger(name):
    return logging.getLogger(name)


def progress_bar(iteration, total, length=100, prefix='Progress:', suffix='Complete', decimals=1, fill='█', print_end="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


def plot_data(data1, label1, data2, label2, filename):
    plt.clf()
    plt.plot(data1, label=label1)
    plt.plot(data2, label=label2)
    plt.legend()
    plt.savefig(filename)
