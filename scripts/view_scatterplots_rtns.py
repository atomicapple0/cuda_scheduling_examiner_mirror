#!/usr/bin/env python3
# This scripts looks through JSON result files and uses matplotlib to display
# scatterplots containing the min, max and arithmetic mean for distributions of
# samples. In order for a distribution to be included in a plot, its "label"
# field must consist of a single number (may be floating-point). As with the
# other scripts, one plot will be created for each "name" in the output files.
import argparse
import copy
import itertools
import glob
import json
import matplotlib.pyplot as plot
import numpy
import sys

def convert_to_float(s):
    """Takes a string s and parses it as a floating-point number. If s can not
    be converted to a float, this returns None instead."""
    to_return = None
    try:
        to_return = float(s)
    except:
        to_return = None
    return to_return

def plugin_summary_values(plugin, times_key):
    """Takes a single plugin results (one parsed output file) and returns
    a list containing 3 elements: [min duration, max duration, mean duration].
    Durations are converted to milliseconds."""
    durations = []
    for t in plugin["times"]:
        if times_key not in t:
            continue
        times = t[times_key]
        i = 0
        while i < len(times):
            duration = times[i + 1] - times[i]
            durations.append(duration)
            i += 2
    minimum = min(durations) * 1000.0
    maximum = max(durations) * 1000.0
    average = numpy.mean(durations) * 1000.0
    return [minimum, maximum, average]

def scenario_to_distribution(scenario):
    """Takes a scenario, mapping numbers to triplets, and re-shapes the data.
    Returns an array of 4 arrays: [[x values], [min y values], [max y values],
    [average y values]]."""
    x_values = []
    for k in scenario:
        x_values.append(k)
    x_values.sort()
    min_y_values = []
    max_y_values = []
    mean_y_values = []
    for k in x_values:
        triplet = scenario[k]
        min_y_values.append(triplet[0])
        max_y_values.append(triplet[1])
        mean_y_values.append(triplet[2])
    return [x_values, min_y_values, max_y_values, mean_y_values]

def add_plot_padding(axes):
    """Takes matplotlib axes, and adds some padding so that lines close to
    edges aren't obscured by tickmarks or the plot border."""
    y_limits = axes.get_ybound()
    y_range = y_limits[1] - y_limits[0]
    y_pad = y_range * 0.05
    x_limits = axes.get_xbound()
    x_range = x_limits[1] - x_limits[0]
    x_pad = x_range * 0.05
    axes.set_ylim(y_limits[0] - y_pad, y_limits[1] + y_pad)
    axes.set_xlim(x_limits[0] - x_pad, x_limits[1] + x_pad)
    #axes.xaxis.set_ticks(numpy.arange(x_limits[0], x_limits[1] + x_pad,
    #    x_range / 5.0))
    axes.xaxis.set_ticks([1, 15, 30, 45, 60])
    axes.yaxis.set_ticks(numpy.arange(y_limits[0], y_limits[1] + y_pad,
        y_range / 5.0))
    return None

def get_marker_styles():
    """ Returns a list of dicts of marker style kwargs. The plot will cycle
    through these for each scenario that's added to the plot. (In practice, I
    only expect to use this script to plot two scenarios at once, so I'm only
    returning two options here for now. """
    base_style = {
        "linestyle": "None",
        "marker": "o",
        "mfc": "k",
        "mec": "k",
        "mew": 0,
        "fillstyle": "full",
        "ms": 4,
        "color": "k",
    }
    style_2 = copy.deepcopy(base_style)
    style_2["marker"] = "s"
    style_2["ms"] = 5
    style_2["mec"] = "0.7"
    style_2["mfc"] = "0.7"
    return [style_2, base_style]

def add_scenario_to_plot(axes, scenario, name, style_dict):
    data = scenario_to_distribution(scenario)
    # data[0] = x vals, data[1] = min, data[2] = max, data[3] = avg
    if "striped" in name:
        name = "SE-distributed"
    else:
        name = "SE-packed"
    axes.plot(data[0], data[3], label=name, **style_dict)
    axes.set_ylabel("Average MM1024 Time (ms)")
    axes.set_xlabel("TPC Partition Size (# of TPCs)")
    legend = plot.legend()
    #legend.draggable()
    legend.set_draggable(True)
    return None

def show_plots(filenames, times_key):
    """ Takes a list of filenames and generates one plot. This differs from the
    hip_plugin_framework script in that it only generates a single plot,
    containing only the average times. It will show one distribution per named
    scenario in the files. """
    # Maps plugin names to plugin data, where the plugin data is a map
    # of X-values to y-value triplets.
    all_scenarios = {}
    counter = 1
    for name in filenames:
        print("Parsing file %d / %d: %s" % (counter, len(filenames), name))
        counter += 1
        with open(name) as f:
            parsed = json.loads(f.read())
            if "label" not in parsed:
                print("Skipping %s: no \"label\" field in file." % (name))
                continue
            if len(parsed["times"]) < 2:
                print("Skipping %s: no recorded times in file." % (name))
                continue
            float_value = convert_to_float(parsed["label"])
            if float_value is None:
                print("Skipping %s: label isn't a number." % (name))
                continue
            summary_values = plugin_summary_values(parsed, times_key)
            name = parsed["scenario_name"]
            if name not in all_scenarios:
                all_scenarios[name] = {}
            all_scenarios[name][float_value] = summary_values

    # Add each scenario to the plot.
    style_cycler = itertools.cycle(get_marker_styles())
    figure = plot.figure()
    figure.canvas.set_window_title("CU partition size vs. MM1024 Time")
    axes = figure.add_subplot(1, 1, 1)
    axes.autoscale(enable=True, axis='both', tight=True)
    for name in all_scenarios:
        add_scenario_to_plot(axes, all_scenarios[name], name,
            next(style_cycler))
    add_plot_padding(axes)
    #plot.subplots_adjust(bottom=0.35)
    plot.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory",
        help="Path prefix of result JSON files.", default='./')
    parser.add_argument("-k", "--times_key",
        help="JSON key name for the time property to be plot.",
        default="execute_times")
    args = parser.parse_args()
    filenames = glob.glob(args.directory + "*.json")
    show_plots(filenames, args.times_key)

