import os
import pickle

import numpy as np
import pandas as pd
from datetime import datetime

# import torch
from tqdm import tqdm

from common.dataloader import get_map_dict
from param_parser import parameter_parser
# import matplotlib.pyplot as plt


def reformat_data(data_path, save_path, dataset="TKY"):
    data = pd.read_csv(data_path)
    month_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_dict = {}
    for i, month in enumerate(month_list):
        if i + 1 < 10:
            month_dict[month] = f'0{i + 1}'
        else:
            month_dict[month] = f'{i + 1}'

    weekday_list = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_dict = {}
    for i, weekday in enumerate(weekday_list):
        weekday_dict[weekday] = i

    print("正在重构原始数据集...")
    filename = 'TKY_reformatted.csv'.replace("TKY", dataset)
    with open(os.path.join(save_path, filename), 'w') as f:
        print('user_id,POI_id,POI_catid,latitude,longitude,day_of_week,norm_in_day_time,time', file=f)
        for user, poi, cat, lat, lon, time in tqdm(zip(list(data["user_id"]), list(data['POI_id']), list(data['POI_catid']),
                                                   list(data['latitude']), list(data['longitude']),
                                                   list(data['UTC_time']))):
            # 过滤签入次数小于 10 的 POIs
            poi_data = data[data['POI_id'] == poi]
            if len(list(poi_data['POI_id'])) < 10:
                continue

            # 过滤签入记录小于 10 的用户
            user_data = data[data["user_id"] == user]
            if len(list(user_data["user_id"])) < 10:
                continue
            time_formatted = f'{time.split(" ")[5]}-{month_dict[time.split(" ")[1]]}-{time.split(" ")[2]} {time.split(" ")[3]}'
            day_of_week = weekday_dict[time.split(" ")[0]]
            hour = time.split(" ")[3].split(":")[0]
            minute = time.split(" ")[3].split(":")[1]
            norm_in_day_time = int(hour) * 2
            if int(minute) > 29:
                norm_in_day_time += 1
            norm_in_day_time = float(norm_in_day_time / 48)
            print(f'{user},{poi},{cat},{lat},{lon},{day_of_week},{norm_in_day_time},{time_formatted}', file=f)


def reformat_sin_data(data_path, save_path):
    data = pd.read_csv(data_path)

    cat_list = set(list(data['POI_catname']))
    cat_dict = dict(zip(cat_list, range(len(cat_list))))

    print("正在重构原始数据集...")
    with open(os.path.join(save_path, "SIN_reformatted.csv"), 'w') as f:
        print('user_id,POI_id,POI_catid,latitude,longitude,day_of_week,norm_in_day_time,time', file=f)
        for user, poi, cat, lat, lon, time in tqdm(zip(list(data["user_id"]), list(data['POI_id']), list(data['POI_catname']),
                                                   list(data['latitude']), list(data['longitude']),
                                                   list(data['UTC_time']))):
            # 过滤签入次数小于 10 的 POIs
            poi_data = data[data['POI_id'] == poi]
            if len(list(poi_data['POI_id'])) < 10:
                continue

            # 过滤签入记录小于 10 的用户
            user_data = data[data["user_id"] == user]
            if len(list(user_data["user_id"])) < 10:
                continue

            cat_id = cat_dict[cat]

            date = datetime.strptime(time.split(" ")[0], '%Y-%m-%d')

            day_of_week = datetime.date(date).weekday()
            hour = time.split(" ")[1].split(":")[0]
            minute = time.split(" ")[1].split(":")[1]
            norm_in_day_time = int(hour) * 2
            if int(minute) > 29:
                norm_in_day_time += 1
            norm_in_day_time = float(norm_in_day_time / 48)
            print(f'{user},{poi},{cat_id},{lat},{lon},{day_of_week},{norm_in_day_time},{time}', file=f)


def create_trajectory(data_path='../../datasets/TKY/TKY_reformatted.csv',
                      save_path='../../datasets/TKY/',
                      dataset="TKY",
                      write_type='w',
                      last_in=0):
    data = pd.read_csv(data_path)
    user_list = list(set(data['user_id']))
    print("\n正在创建用户轨迹...")
    filename = 'TKY_reformatted_trajectory.csv'.replace("TKY", dataset)
    with open(os.path.join(save_path, filename), write_type) as f:
        if write_type == 'w':
            print('user_id,POI_id,POI_catid,latitude,longitude,day_of_week,norm_in_day_time,time,trajectory_id', file=f)
        for user in tqdm(user_list):
            if user < last_in:
                continue
            user_data = data[data['user_id'] == user]
            user_time_list = []
            for time in list(user_data['time']):
                # 数据集中有一条错误数据
                if time == '[]':
                    time = "2013-04-08 12:34:40"
                try:
                    time_formatted = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
                    user_time_list.append(time_formatted)
                except Exception:
                    print(time, user)
                    time = "2013-04-08 12:34:40"
                    time_formatted = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
                    user_time_list.append(time_formatted)

            user_trajectory = []
            j = 1
            trajectory_id_data = []
            for i, time in enumerate(user_time_list):
                trajectory_id = f"{user}_{j}"
                trajectory_id_data.append([
                            user, list(user_data["POI_id"])[i], list(user_data["POI_catid"])[i],
                            list(user_data["latitude"])[i], list(user_data["longitude"])[i],
                            list(user_data["day_of_week"])[i], list(user_data["norm_in_day_time"])[i],
                            time, trajectory_id]
                )
                if i < len(user_time_list) - 1:
                    if user_time_list[i + 1].month == user_time_list[i].month:
                        time_distance = (user_time_list[i + 1].day - user_time_list[i].day) * 24 \
                                         + user_time_list[i + 1].hour - user_time_list[i].hour
                        if time_distance > 24:
                            user_trajectory.append(trajectory_id_data)
                            trajectory_id_data = []
                            j += 1
                    else:
                        user_trajectory.append(trajectory_id_data)
                        trajectory_id_data = []
                        j += 1

            for trajectory_id_data in user_trajectory:
                if len(trajectory_id_data) < 1:
                    continue
                else:
                    for trajectory_data in trajectory_id_data:
                        print(f'{trajectory_data[0]},{trajectory_data[1]},{trajectory_data[2]},{trajectory_data[3]},'
                              f'{trajectory_data[4]},{trajectory_data[5]},{trajectory_data[6]},{trajectory_data[7]},'
                              f'{trajectory_data[8]}', file=f)


def sort_trajectory_by_time(data_path='../datasets/TKY/TKY_reformatted_trajectory.csv',
                            save_path='../datasets/TKY/',
                            dataset="TKY"):
    print("\nSorting users' trajectories with checkin time information...")
    trajectory_data = pd.read_csv(data_path)
    trajectory_data.sort_values(by=["time"], inplace=True)

    filename = 'TKY_all.csv'.replace("TKY", dataset)
    trajectory_data.to_csv(os.path.join(save_path, filename), index=False)


def dataset_split(data_path='../datasets/TKY/TKY_all.csv',
                  save_path='../datasets/TKY/',
                  dataset="TKY"):
    data = pd.read_csv(data_path)

    print("\n正在划分数据集...")
    data_len = len(list(data['user_id']))
    train_len = int(data_len * 0.8)
    val_len = int(data_len * 0.9)
    train_data = data[:train_len]
    val_data = data[train_len:val_len]
    test_data = data[val_len:]

    print("正在存储训练集...")
    filename = 'TKY_train.csv'.replace("TKY", dataset)
    train_data.to_csv(os.path.join(save_path, filename), index=False)

    print("\n正在存储验证集...")
    filename = 'TKY_val.csv'.replace("TKY", dataset)
    val_data.to_csv(os.path.join(save_path, filename), index=False)

    print("\n正在存储测试集...")
    filename = 'TKY_test.csv'.replace("TKY", dataset)
    test_data.to_csv(os.path.join(save_path, filename), index=False)


def dataset_split_():
    data = pd.read_csv("../datasets/Foursquare/data/Foursquare_all_100.csv")

    data_len = len(list(data['user_id']))
    new_len = int(data_len * 0.125)
    train_data = data[:new_len]
    train_data.to_csv("../datasets/Foursquare/Foursquare_all.csv", index=False)

# NYC-F 数据集中存在部分重复数据，用此函数可以清除重复数据
def remove_duplicate(data_path):
    data = pd.read_csv(data_path)
    deduplicated_data = data.drop_duplicates()
    deduplicated_data.to_csv('../datasets/NYC-F/NYC-F-all.csv', index=False)


if __name__ == '__main__':
    parsers = parameter_parser()
    # reformat_sin_data('../datasets/SIN/SIN.csv', "../datasets/SIN/")
    # create_trajectory('../datasets/Foursquare_City/Foursquare.csv', "../datasets/Foursquare_City/", dataset="Foursquare_City",
    #                   write_type='w', last_in=0)
    # sort_trajectory_by_time(data_path='../datasets/Foursquare_City/data/Foursquare_reformatted_trajectory.csv',
    #                         save_path='../datasets/Foursquare_City/', dataset="Foursquare_City")
    # dataset_split_()
    # dataset_split('../datasets/Foursquare/Foursquare_all.csv', "../datasets/Foursquare/", dataset="Foursquare")
    # parsers.data_train = "../datasets/SIN/SIN_train.csv"

    # date_time_feature_analyse(parsers)
    # print("=====" * 30)
    # date_day_feature_analyse(parsers)
    # date_user_feature_analyse(parsers)

    # remove_duplicate("../datasets/NYC-F/NYC_all.csv")
    dataset_split('../datasets/NYC-F/NYC-F-all.csv', "../datasets/NYC-F/", dataset="NYC-F")
