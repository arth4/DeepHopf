import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm.auto import tqdm

def plot_multiple(Series, **kwargs):
    for series in Series:
        plt.plot(series)
    plt.xlabel(kwargs.get("xlabel", ""))
    plt.ylabel(kwargs.get("ylabel", ""))
    plt.title(kwargs.get("title", ""))
    plt.legend(kwargs.get("labels", []))
    
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        k, v = values.split("=")
        if getattr(namespace, self.dest) is None:
            setattr(namespace, self.dest, {})
        d = getattr(namespace, self.dest)
        d[k] = self.try_eval(v)


    def try_eval(self, v):
        try:
            v = eval(v)
        except:
            pass
        return v

class ProgressBar(tqdm):
    pass
