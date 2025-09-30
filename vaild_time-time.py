import xarray as xr

def rename_time_and_save(filename, output_filename):
    # 打开 NetCDF 文件作为一个 xarray Dataset
    ds = xr.open_dataset(filename)

    # 如果存在名为 'valid_time' 的坐标变量，则重命名为 'time'
    if 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
        print("Renamed 'valid_time' to 'time'.")
    else:
        print("'valid_time' not found. Assuming 'time' is already correct.")

    # 保存为新的 NetCDF 文件
    ds.to_netcdf(output_filename)
    print(f"Saved the modified dataset to {output_filename}")

# 指定输入和输出文件路径
input_filename = r'D:\zyh\dataset\SST\download\sst_2014_2023.nc'
output_filename = r'D:\zyh\dataset\SST\download\sst_2014_2023_renamed_time.nc'

# 调用函数以重命名并保存
rename_time_and_save(input_filename, output_filename)