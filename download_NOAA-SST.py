# -*- coding:utf-8 -*-
# this file is used to download SST data massively from NOAA for paleoclimate research
# 2025-01-06
# author: ming
# using BeautifulSoup and concurrent.futures for multithreading

import re
from bs4 import BeautifulSoup
import urllib.request
import ssl
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta


# 忽略 SSL 证书验证（仅用于 HTTPS 请求）
ssl._create_default_https_context = ssl._create_unverified_context

base_url = 'https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/'
output_dir_base = r"D:\zyh\dataset\OSSIT"


def get_month_urls(start_date, end_date):
    """Generate URLs for each month within the specified range."""
    current = start_date
    urls = []
    while current <= end_date:
        folder_name = current.strftime('%Y%m')
        url = f'{base_url}{folder_name}/'
        urls.append((url, folder_name))
        # Move to the next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return urls


def fetch_file_links(url):
    """Fetch all .nc file links from a given month URL."""
    try:
        with urllib.request.urlopen(url) as response:
            html = response.read().decode('utf-8')  # decode with utf-8 instead of ascii
            soup = BeautifulSoup(html, 'html.parser')
            links = soup.find_all(href=re.compile(r".nc$"))
            return [urllib.parse.urljoin(url, link.get('href')) for link in links]
    except Exception as e:
        print(f"Failed to fetch the page {url}: {e}")
        return []


def download_file(url, output_dir):
    """Download a single file from its URL into the specified directory."""
    try:
        file_name = os.path.join(output_dir, url.split('/')[-1])
        urllib.request.urlretrieve(url, file_name)
        return f'Successfully downloaded: {file_name}'
    except Exception as e:
        return f'Failed to download {url}: {e}'


if __name__ == '__main__':
    start_date = date(2024, 12, 1)
    end_date = date(2024, 12, 31)

    month_urls = get_month_urls(start_date, end_date)

    for month_url, folder_name in month_urls:
        print(f'Processing {folder_name}')
        # 创建每个月份对应的子目录
        output_dir = os.path.join(output_dir_base, folder_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f'Fetching URLs from {month_url}')
        file_urls = fetch_file_links(month_url)
        print(f'Found {len(file_urls)} files in {month_url}')

        print(f'Starting downloads for {folder_name}...')
        with ThreadPoolExecutor(max_workers=31) as executor:
            futures = [executor.submit(download_file, url, output_dir) for url in file_urls]
            for future in as_completed(futures):
                print(future.result())

    print('All downloads are complete.')