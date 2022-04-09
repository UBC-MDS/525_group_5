# Milestone 2

In this milestone, we will be migrating our analysis to the **AWS cloud**. The process is as such:  

(1) Set up collaborative environment via EC2 instance with JupyterHub, set up S3 bucket  
(2) Migrate data from Milestone 1 to S3  
(3) Wrangle data in preparation for Machine Learning

## 1. Setup Collaborative Environment

### 1.1 Setup EC2 instance

![](img/EC2_screenshot.PNG)

### 1.2. Setup JupyterHub

 ![](img/jupyterhub_screenshot.PNG)

### 1.3 Setup the server

![](img/server_screenshot.PNG)

## 2. Migrate data from Milestone 1 to S3

### 2.1 Setup S3 bucket and move data

![](img/S3_bucket.PNG)

## 3. Wrangle Data

#### Installing packages


```python
# !pip install pandas
```


```python
# !pip install pyarrow
```


```python
# !pip install s3fs
```

#### Reading parquet file from S3 bucket


```python
import pandas as pd
aws_credentials = {"key" : "ASIATPTZJGVKER44PNNH",
                "secret": "CJyy69lmWSDqyJTef2LU1kTNOLE8Cg7MVHjds7Lx",
                "token": "FwoGZXIvYXdzEFgaDA5nyZSZt+cg1eB6oCLEAbgmthG3gMBUl6sU4v5E0KLLdD7NRSu3u/RN6miN/xbdap9C+LLT/tgoRKFOWGRwmiOZXSIdqw5duL0ygTp6EGupFjhghF0yFeEaXEXHJRXfpG0WiK0bcLy/w1FpZWHFaJzJsX1QRuoWXU2IYy7IxaGt5mYzQG2gLPCjC0ImdbEVuRRzB5DtKIkRJ9uOHTDmmgoeB/ahndy24QeZ9ZYOB4lTZWDfRe7drck/WZEmwlnVo+Yd87Oz4XnuJx5JsB7t42ZiwyMo7PnCkgYyLaAGNnGwbkPGhgif+7rSKh0wENigsUx+CgFv1y+XKmr92FgZmkM6IB0N11GiDw=="}

combined_df = pd.read_parquet('s3://mds-s3-5/combined_model_data_parti.parquet',
                     storage_options=aws_credentials)
combined_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>lat_min</th>
      <th>lat_max</th>
      <th>lon_min</th>
      <th>lon_max</th>
      <th>rain (mm/day)</th>
      <th>model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1889-01-01 12:00:00</td>
      <td>-36.25</td>
      <td>-35.0</td>
      <td>140.625</td>
      <td>142.5</td>
      <td>3.293256e-13</td>
      <td>ACCESS-CM2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1889-01-02 12:00:00</td>
      <td>-36.25</td>
      <td>-35.0</td>
      <td>140.625</td>
      <td>142.5</td>
      <td>0.000000e+00</td>
      <td>ACCESS-CM2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1889-01-03 12:00:00</td>
      <td>-36.25</td>
      <td>-35.0</td>
      <td>140.625</td>
      <td>142.5</td>
      <td>0.000000e+00</td>
      <td>ACCESS-CM2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1889-01-04 12:00:00</td>
      <td>-36.25</td>
      <td>-35.0</td>
      <td>140.625</td>
      <td>142.5</td>
      <td>0.000000e+00</td>
      <td>ACCESS-CM2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1889-01-05 12:00:00</td>
      <td>-36.25</td>
      <td>-35.0</td>
      <td>140.625</td>
      <td>142.5</td>
      <td>1.047658e-02</td>
      <td>ACCESS-CM2</td>
    </tr>
  </tbody>
</table>
</div>



#### Filtering Sydney lat-lon


```python
combined_df['valid_min_lat'] = combined_df['lat_min'] < -33.86
combined_df['valid_max_lat'] = combined_df['lat_max'] > -33.86
combined_df['valid_min_lon'] = combined_df['lon_min'] < 151.21
combined_df['valid_max_lon'] = combined_df['lon_max'] > 151.21

combined_df = combined_df[combined_df['valid_min_lat'] == True][combined_df['valid_max_lat'] == True][combined_df['valid_min_lon'] == True][combined_df['valid_max_lon'] == True]

combined_df.head()

```

    /var/folders/nt/dj9wy9hn23v5kt5rnqc8zlw80000gn/T/ipykernel_12377/3738855013.py:6: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      combined_df = combined_df[combined_df['valid_min_lat'] == True][combined_df['valid_max_lat'] == True][combined_df['valid_min_lon'] == True][combined_df['valid_max_lon'] == True]





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>lat_min</th>
      <th>lat_max</th>
      <th>lon_min</th>
      <th>lon_max</th>
      <th>rain (mm/day)</th>
      <th>model</th>
      <th>valid_min_lat</th>
      <th>valid_max_lat</th>
      <th>valid_min_lon</th>
      <th>valid_max_lon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>552240</th>
      <td>1889-01-01 12:00:00</td>
      <td>-35.0</td>
      <td>-33.75</td>
      <td>150.0</td>
      <td>151.875</td>
      <td>0.040427</td>
      <td>ACCESS-CM2</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>552241</th>
      <td>1889-01-02 12:00:00</td>
      <td>-35.0</td>
      <td>-33.75</td>
      <td>150.0</td>
      <td>151.875</td>
      <td>0.073777</td>
      <td>ACCESS-CM2</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>552242</th>
      <td>1889-01-03 12:00:00</td>
      <td>-35.0</td>
      <td>-33.75</td>
      <td>150.0</td>
      <td>151.875</td>
      <td>0.232656</td>
      <td>ACCESS-CM2</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>552243</th>
      <td>1889-01-04 12:00:00</td>
      <td>-35.0</td>
      <td>-33.75</td>
      <td>150.0</td>
      <td>151.875</td>
      <td>0.911319</td>
      <td>ACCESS-CM2</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>552244</th>
      <td>1889-01-05 12:00:00</td>
      <td>-35.0</td>
      <td>-33.75</td>
      <td>150.0</td>
      <td>151.875</td>
      <td>0.698013</td>
      <td>ACCESS-CM2</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



#### Keep relevant columns and adjusting 'time' column to keep only date


```python
combined_df = combined_df[['time', 'rain (mm/day)', 'model']]
```


```python
combined_dates = combined_df['time'].tolist()
only_dates = []
for v in combined_dates:
    only_dates.append(str(v).split(" ")[0])
```


```python
combined_df['time'] = only_dates
combined_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>rain (mm/day)</th>
      <th>model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>552240</th>
      <td>1889-01-01</td>
      <td>0.040427</td>
      <td>ACCESS-CM2</td>
    </tr>
    <tr>
      <th>552241</th>
      <td>1889-01-02</td>
      <td>0.073777</td>
      <td>ACCESS-CM2</td>
    </tr>
    <tr>
      <th>552242</th>
      <td>1889-01-03</td>
      <td>0.232656</td>
      <td>ACCESS-CM2</td>
    </tr>
    <tr>
      <th>552243</th>
      <td>1889-01-04</td>
      <td>0.911319</td>
      <td>ACCESS-CM2</td>
    </tr>
    <tr>
      <th>552244</th>
      <td>1889-01-05</td>
      <td>0.698013</td>
      <td>ACCESS-CM2</td>
    </tr>
  </tbody>
</table>
</div>



#### Reading observed data from S3 bucket


```python
obs = pd.read_csv('s3://mds-s3-5/observed_daily_rainfall_SYD.csv', storage_options=aws_credentials)
obs.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>rain (mm/day)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1889-01-01</td>
      <td>0.006612</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1889-01-02</td>
      <td>0.090422</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1889-01-03</td>
      <td>1.401452</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1889-01-04</td>
      <td>14.869798</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1889-01-05</td>
      <td>0.467628</td>
    </tr>
  </tbody>
</table>
</div>



#### Added 'model' column and adjust 'time' column to keep only date


```python
observed_dates = obs['time'].tolist()
obs_dates = []
for v in observed_dates:
    obs_dates.append(str(v).split(" ")[0])

obs['time'] = obs_dates
obs['model'] = 'observed_rainfall'

obs.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>rain (mm/day)</th>
      <th>model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1889-01-01</td>
      <td>0.006612</td>
      <td>observed_rainfall</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1889-01-02</td>
      <td>0.090422</td>
      <td>observed_rainfall</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1889-01-03</td>
      <td>1.401452</td>
      <td>observed_rainfall</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1889-01-04</td>
      <td>14.869798</td>
      <td>observed_rainfall</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1889-01-05</td>
      <td>0.467628</td>
      <td>observed_rainfall</td>
    </tr>
  </tbody>
</table>
</div>



#### Combine both dataframes


```python
combined_df = pd.concat((combined_df, obs))
```

#### Pivot dataframe to get the desired output


```python
combined_df = combined_df.pivot(index='time', columns='model', values='rain (mm/day)')
combined_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>model</th>
      <th>ACCESS-CM2</th>
      <th>ACCESS-ESM1-5</th>
      <th>AWI-ESM-1-1-LR</th>
      <th>BCC-CSM2-MR</th>
      <th>BCC-ESM1</th>
      <th>CMCC-CM2-HR4</th>
      <th>CMCC-CM2-SR5</th>
      <th>CMCC-ESM2</th>
      <th>CanESM5</th>
      <th>EC-Earth3-Veg-LR</th>
      <th>...</th>
      <th>MPI-ESM-1-2-HAM</th>
      <th>MPI-ESM1-2-HR</th>
      <th>MPI-ESM1-2-LR</th>
      <th>MRI-ESM2-0</th>
      <th>NESM3</th>
      <th>NorESM2-LM</th>
      <th>NorESM2-MM</th>
      <th>SAM0-UNICON</th>
      <th>TaiESM1</th>
      <th>observed_rainfall</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1889-01-01</th>
      <td>0.040427</td>
      <td>1.814552</td>
      <td>35.579336</td>
      <td>4.268112e+00</td>
      <td>1.107466e-03</td>
      <td>11.410537</td>
      <td>3.322009e-08</td>
      <td>2.668800</td>
      <td>1.321215</td>
      <td>1.515293</td>
      <td>...</td>
      <td>4.244226e-13</td>
      <td>1.390174e-13</td>
      <td>6.537884e-05</td>
      <td>3.445495e-06</td>
      <td>1.576096e+01</td>
      <td>4.759651e-05</td>
      <td>2.451075</td>
      <td>0.221324</td>
      <td>2.257933</td>
      <td>0.006612</td>
    </tr>
    <tr>
      <th>1889-01-02</th>
      <td>0.073777</td>
      <td>0.303965</td>
      <td>4.596520</td>
      <td>1.190141e+00</td>
      <td>1.015323e-04</td>
      <td>4.014984</td>
      <td>1.312700e+00</td>
      <td>0.946211</td>
      <td>2.788724</td>
      <td>4.771375</td>
      <td>...</td>
      <td>4.409552e+00</td>
      <td>1.222283e-01</td>
      <td>1.049131e-13</td>
      <td>4.791993e-09</td>
      <td>3.675510e-01</td>
      <td>4.350863e-01</td>
      <td>0.477231</td>
      <td>3.757179</td>
      <td>2.287381</td>
      <td>0.090422</td>
    </tr>
    <tr>
      <th>1889-01-03</th>
      <td>0.232656</td>
      <td>0.019976</td>
      <td>5.927467</td>
      <td>1.003845e-09</td>
      <td>1.760345e-05</td>
      <td>9.660565</td>
      <td>9.103720e+00</td>
      <td>0.431999</td>
      <td>0.003672</td>
      <td>4.233980</td>
      <td>...</td>
      <td>2.269300e-01</td>
      <td>3.762301e-01</td>
      <td>9.758706e-14</td>
      <td>6.912302e-01</td>
      <td>1.562869e-01</td>
      <td>9.561101e+00</td>
      <td>0.023083</td>
      <td>0.253357</td>
      <td>1.199909</td>
      <td>1.401452</td>
    </tr>
    <tr>
      <th>1889-01-04</th>
      <td>0.911319</td>
      <td>13.623777</td>
      <td>8.029624</td>
      <td>8.225225e-02</td>
      <td>1.808932e-01</td>
      <td>3.951528</td>
      <td>1.317160e+01</td>
      <td>0.368693</td>
      <td>0.013578</td>
      <td>15.252495</td>
      <td>...</td>
      <td>2.344586e-02</td>
      <td>4.214019e-01</td>
      <td>7.060915e-03</td>
      <td>3.835721e-02</td>
      <td>2.472226e-07</td>
      <td>5.301038e-01</td>
      <td>0.002699</td>
      <td>2.185454</td>
      <td>2.106737</td>
      <td>14.869798</td>
    </tr>
    <tr>
      <th>1889-01-05</th>
      <td>0.698013</td>
      <td>0.021048</td>
      <td>2.132686</td>
      <td>2.496841e+00</td>
      <td>4.708019e-09</td>
      <td>2.766362</td>
      <td>1.822940e+01</td>
      <td>0.339267</td>
      <td>0.002468</td>
      <td>11.920356</td>
      <td>...</td>
      <td>4.270161e-13</td>
      <td>1.879692e-01</td>
      <td>4.504985e+00</td>
      <td>3.506923e-07</td>
      <td>1.949792e-13</td>
      <td>1.460928e-10</td>
      <td>0.001026</td>
      <td>2.766507</td>
      <td>1.763335</td>
      <td>0.467628</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
combined_df.shape
```




    (46020, 26)



#### Save data to S3 bucket


```python
combined_df.to_csv('s3://mds-s3-5/output/ml_data_SYD.csv', storage_options=aws_credentials)
```

![](img/output_file_S3.PNG)


```python

```
