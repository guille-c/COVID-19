from flask import Flask, render_template, redirect, url_for, request
import pandas as pd
import numpy as np
import charts
app = Flask(__name__)

# Reading data from csv
url = "https://raw.githubusercontent.com/guille-c/COVID-19/master/data/COVID_Chile_Regiones.csv"
df = pd.read_csv (url)
df = df.dropna(subset = ["Region"])
df["Fecha_dt"] = pd.to_datetime (df["Fecha"])
df["Fecha_dt"][df["Fecha_dt"] >= '2020-03-18'] -=  pd.Timedelta(hours=12)

dic = dict()
for k, name in enumerate(df['Region'].unique()):
    dic[k+1] = name


@app.route('/results/<id>/<e0>/<r0>/<n>', methods=['GET', 'POST'])
def results(id, e0, r0, n):
    region = dic[int(id)]
    df_reg = df[df.Region == region]
    df_c = df_reg[df_reg["Contagiados"] > 0]
    df_c = df_c.groupby("Fecha_dt", as_index=False).sum()

    i_data = df_c["Contagiados"].values
    x_times = (pd.DataFrame(df_c["Fecha_dt"] - df_c["Fecha_dt"].iloc[0])/np.timedelta64(1, 'D')).values.flatten()
    x_times = x_times[i_data > 50]
    i_ini = len(i_data) - len(x_times)
    i_data = i_data[i_data > 50]

    script_seir, div_seir = charts.get_SEIR_pred(x_times, i_data,
                                                 float(e0), float(r0),
                                                 int(n))

    script_i, div_i  = charts.get_infectados(x_times,
                                      df_c['Contagiados'].values,
                                      df_reg['Fecha'].values)


    return render_template('results.html',
                            region=region,
                            script_i=script_i,
                            div_i=div_i,
                            script_seir=script_seir,
                            div_seir=div_seir)



@app.route('/', methods = ['GET', 'POST'])
def main():
    if request.method == 'POST':
        region_seleccionada = request.form.get('region_list')

        n = request.form.get('n')
        r = request.form.get('r')
        e = request.form.get('e')
        return redirect(
        url_for('results',id=region_seleccionada,e0=e,r0=r,n=n))

    return render_template('index.html', regiones=dic)

if __name__ == '__main__':
   app.run(host='0.0.0.0')
