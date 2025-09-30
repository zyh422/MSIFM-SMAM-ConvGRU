import xarray as xr

def read_nc_variables_with_xarray(filename):
    # 打开 NetCDF 文件作为一个 xarray Dataset
    ds = xr.open_dataset(filename)

    # 遍历每个变量并打印信息
    for var_name in ds.data_vars:
        variable = ds[var_name]
        print(f"Variable Name: {var_name}")
        print(f"\tDimensions: {variable.dims}")
        print(f"\tData Type: {variable.dtype}")
        print(f"\tAttributes: {variable.attrs}")
        print()

    try:
        # 提取时间范围
        time_range = (ds['time'].values[0], ds['time'].values[-1])
        print(f"Time Range: {time_range}")
    except KeyError as e:
        print(f"Warning: Could not find time variable. Error: {e}")

    # 提取经纬度范围
    lat_range = (ds['lat'].min().values, ds['lat'].max().values)
    lon_range = (ds['lon'].min().values, ds['lon'].max().values)
    print(f"Latitude Range: {lat_range}")
    print(f"Longitude Range: {lon_range}")
    # 打印 thetao 变量的形状
    if 'thetao' in ds:
        thetao = ds['thetao']
        print(f"Shape of thetao: {thetao.shape}")
        print(f"Dimensions of thetao: {thetao.dims}")
        print(f"Data Type of thetao: {thetao.dtype}")
        print(f"Attributes of thetao: {thetao.attrs}")
    else:
        print("Variable 'thetao' not found in the dataset.")

# filename = '路径到你的NetCDF文件'

filename = r'D:\zyh\dataset\OISST\oisst_daily_1982_2023.nc'

read_nc_variables_with_xarray(filename)

