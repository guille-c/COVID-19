from bokeh.models import ColumnDataSource, PreText, Select
from bokeh.plotting import figure
from bokeh.embed import components, json_item
from bokeh.layouts import column
import numpy as np
import sys
import static.src.SEIR as SEIR

def get_infectados(x, y, date):
    source = ColumnDataSource(data=dict(
                x=x,
                y=y,
                date=date))
    TOOLTIPS = [
        ("index", "$index"),
        ("contagios", "@y"),
        ("fecha", "@date"),
    ]
    tools = 'pan,wheel_zoom,xbox_select,reset'

    p1 = figure(tools=tools,
                tooltips=TOOLTIPS,
                plot_width=600, plot_height=300,
                title="Historial de Infectados")
    p1.scatter('x', 'y', size=10, color="red", alpha=0.5, source=source)
    p1.line('x', 'y', source=source)
    p1.background_fill_alpha = 0.3

    return components(p1)

def get_SEIR_pred(x_times, i_data, e0, r0, n):
    i0 = i_data[0]
    s0 = n - i0

    RMSE, beta, sigma, gamma = SEIR.GridSearchSEIR (x_times,
                                               i_data,
                                               s0,
                                               e0,
                                               i0,
                                               r0,
                                               backward = True)

    x_times_long = np.arange(x_times[0], 120, 0.5)

    s_cl, e_cl, i_cl, r_cl = SEIR.SEIR_backward(x_times_long,
                                           s0,
                                           e0,
                                           i0,
                                           r0,
                                           beta,
                                           sigma,
                                           gamma)

    source = ColumnDataSource(data=dict(
                x=x_times_long,
                y=i_cl))

    TOOLTIPS = [
        ("index", "$index"),
    ]
    tools = 'pan,wheel_zoom,xbox_select,reset'

    p = figure(tools=tools,
               tooltips=TOOLTIPS,
               plot_width=600, plot_height=300,
               title="Modelo SEIR (a0: {} - r0: {} - poblacion:{})".format(e0, r0, n))
    p.scatter('x', 'y', size=10, color="red", alpha=0.5, source=source)
    p.line('x', 'y', source=source)
    p.background_fill_alpha = 0.3

    return components(p)
