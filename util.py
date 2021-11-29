import logging
import pathlib
import matplotlib.pyplot as plt
from datetime import datetime
import time
import numpy as np

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


def progress_bar_with_time(done, total, start):
    now = datetime.now()
    left = (total - done) * (now - start) / done
    remaining_time = time.strftime('%H:%M:%S', time.gmtime(left.total_seconds()))
    progress_bar(done, total, length=50, prefix='Training:', suffix=remaining_time)

def plot_data(data_with_label, filename):
    plt.clf()
    for (data, label) in data_with_label:
        plt.plot(data, label=label, linewidth=3.0)
    plt.legend()
    plt.savefig(filename)


def plot_data(data1, label1, data2, label2, filename):
    plt.clf()
    plt.plot(data1, label=label1, linewidth=1.1)
    plt.plot(data2, label=label2, linewidth=1.1)
    plt.legend()
    plt.savefig(filename)

def plot_dice(data_with_label, filename):
    plt.clf()
    plt.yticks(np.arange(0.0, 1.1, 0.2))
    for (data, label) in data_with_label:
        plt.plot(data, label=label, linewidth=1.1)
    plt.grid()
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, len(data_with_label[0][0])-1])
    plt.legend()
    plt.savefig(filename)


def boxplot(data, labels, filename):
    red_square = dict(markerfacecolor='r', marker='s')

    fig1, ax1 = plt.subplots()
    ax1.boxplot(data, flierprops=red_square)
    ax1.set_xticklabels(labels)
    plt.ylim([0.0, 1.0])
    plt.ylabel('Dice')
    plt.grid(color='w')
    ax1.set_facecolor((204.0/255.0, 247.0/255.0, 230.0/255.0))
    plt.legend()
    plt.savefig(filename)