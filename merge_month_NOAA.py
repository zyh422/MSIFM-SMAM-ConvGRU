import os
import xarray as xr
from glob import glob
import numpy as np

# 定义路径
base_dir = r'D:\zyh\dataset\OISST'

# 初始化一个空列表用于存储每个月的平均 SST 数据集
monthly_averages = []

# 遍历年份（1981-2023）
for year in range(1982, 2024):
    for month in range(1, 13):
        # 构建每个月份的路径
        month_dir = os.path.join(base_dir, f'{year:04d}{month:02d}')

        if not os.path.exists(month_dir):
            print(f"目录不存在: {month_dir}")
            continue

        # 查找该月份下的所有 .nc 文件
        nc_files = glob(os.path.join(month_dir, '*.nc'))

        if not nc_files:
            print(f"没有找到 .nc 文件: {month_dir}")
            continue

        # 打开并读取所有文件
        ds_list = [xr.open_dataset(file) for file in nc_files]

        # 检查是否有数据集为空或缺少 'sst' 变量
        ds_list = [ds for ds in ds_list if 'sst' in ds.variables and not ds['sst'].isnull().all()]

        if not ds_list:
            print(f"没有有效的 'sst' 数据: {month_dir}")
            continue

        # 将所有数据集合并为一个
        combined_ds = xr.concat(ds_list, dim='time')

        # 计算该月份的平均 SST
        monthly_avg = combined_ds['sst'].mean(dim='time', skipna=True)

        # 创建一个新的数据集来保存这个月的平均 SST
        avg_ds = xr.Dataset(
            {'sst': (('zlev', 'lat', 'lon'), monthly_avg.values)},
            coords={
                'time': [f'{year}-{month:02d}'],  # 添加时间坐标
                'zlev': combined_ds['zlev'],
                'lat': combined_ds['lat'],
                'lon': combined_ds['lon']
            }
        )

        # 添加到每月平均值列表中
        monthly_averages.append(avg_ds)

# 将所有的月平均值拼接成一个数据集
final_ds = xr.concat(monthly_averages, dim='time')

# 设置时间维度为日期类型
final_ds['time'] = xr.cftime_range(start='1982-01-01', periods=len(final_ds.time), freq='MS')

# 保存最终的数据集到一个新的 NetCDF 文件
output_filename = r'D:\zyh\dataset\OISST\oisst_monthly_1982_2023.nc'
final_ds.to_netcdf(output_filename)

print(f"已保存至 {output_filename}")