import matplotlib.pyplot as plt
import matplotlib
def set_matplotlib_params():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    matplotlib.rcParams['font.size'] = 13
    matplotlib.rcParams['axes.linewidth'] = 2
    matplotlib.rcParams['axes.unicode_minus'] = False