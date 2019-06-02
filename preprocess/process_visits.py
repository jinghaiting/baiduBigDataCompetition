import datetime
from datetime import datetime as dt
import os
from tqdm import tqdm
import numpy as np
from paths import train_visits_274_npy, train_visits_224_npy, train_ids_npy, train_visit_dir, train_visits_origin_npy, \
    test_ids_npy, test_visit_dir, test_visits_origin_npy, test_visits_274_npy, test_visits_224_npy

holidays = ('20190128', '20190129', '20190130', '20190131', '20190201', '20190202', '20190203',
            '20190204', '20190205', '20190206', '20190207', '20190208', '20190209', '20190210',
            '20181001', '20181002', '20181003', '20181004', '20181005', '20181006', '20181007')
date2holiday_weekday = {}
date2holiday_weekend = {}

# 用字典查询代替类型转换，可以减少一部分计算时间
date2position = {}
datestr2dateint = {}
str2int = {}
for i in range(24):
    str2int[str(i).zfill(2)] = i

# 访问记录内的时间从2018年10月1日起，共182天
# 将日期按日历排列
for i in range(182):
    date = datetime.date(day=1, month=10, year=2018) + datetime.timedelta(days=i)
    date = date.__str__().replace("-", "")
    date_int = int(date)
    date2position[date_int] = [i % 7, i // 7]
    datestr2dateint[str(date_int)] = date_int

    is_holiday = 1 if date in holidays else 0
    weekday = dt.strptime(date, "%Y%m%d").weekday()
    is_weekend = 1 if weekday > 4 else 0
    date2holiday_weekday[date] = [is_holiday, weekday]
    date2holiday_weekend[date] = [is_holiday, is_weekend]


def process_txt(lines):
    visit_origin = np.zeros((26, 24, 7))
    visit_274 = np.zeros((2, 7, 4))
    visit_224 = np.zeros((2, 2, 4))
    for line in lines:
        line = line.split()[1]  # 默认空格分
        # temp = []
        # for item in line.split(','):
        #     temp.append([item[0:8], item[9:].split("|")])
        temp = [[item[0:8], item[9:].split("|")] for item in line.split(',')]
        for date, hour_lst in temp:
            # x - 第几天
            # y - 第几周
            # z - 几点钟
            # value - 到访的总人数
            x, y = date2position[datestr2dateint[date]]
            is_holiday, weekday = date2holiday_weekday[date]
            is_holiday, is_weekend = date2holiday_weekend[date]
            for hour in hour_lst:  # 统计到访的总人数
                visit_origin[y][str2int[hour]][x] += 1
                visit_274[is_holiday][weekday][int(hour) // 6] += 1
                visit_224[is_holiday][is_weekend][int(hour) // 6] += 1
    visit_274[1] /= 3.0
    visit_274[0] /= 23.0
    visit_224[1] /= 3.0
    visit_224[0] /= 23.0
    visit_224[:, 0, :] /= 5.0
    visit_224[:, 1, :] /= 2.0

    visit = (visit_origin, visit_274, visit_224)
    return visit


def load(ids_npy, visit_dir):
    visits_origin = []
    visits_274 = []
    visits_224 = []
    ids = np.load(ids_npy)
    for id in tqdm(ids):
        txt_name = '%s.txt' % id  # txt_name = id + '.txt'
        txt_path = os.path.join(visit_dir, txt_name)
        with open(txt_path) as f:
            lines = f.readlines()
            visit = process_txt(lines)
            visits_origin.append(visit[0])
            visits_274.append(visit[1])
            visits_224.append(visit[2])

    visits_origin = np.stack(visits_origin)
    visits_274 = np.stack(visits_274)
    visits_224 = np.stack(visits_224)

    visits = (visits_origin, visits_274, visits_224)
    return visits


def load_visits(ids_npy, visit_dir, visits_origin_npy, visits_274_npy, visits_224_npy):
    if os.path.exists(visits_origin_npy) and os.path.exists(visits_274_npy) and os.path.exists(visits_224_npy):
        visits_origin = np.load(visits_origin_npy)
        visits_274 = np.load(visits_274_npy)
        visits_224 = np.load(visits_224_npy)
        visits = (visits_origin, visits_274, visits_224)

    else:
        visits = load(ids_npy, visit_dir)
        np.save(visits_origin_npy, visits[0])
        np.save(visits_274_npy, visits[1])
        np.save(visits_224_npy, visits[2])

    return visits


def main():
    train_args = dict(
        ids_npy=train_ids_npy,
        visit_dir=train_visit_dir,
        visits_origin_npy=train_visits_origin_npy,
        visits_274_npy=train_visits_274_npy,
        visits_224_npy=train_visits_224_npy
    )
    test_args = dict(
        ids_npy=test_ids_npy,
        visit_dir=test_visit_dir,
        visits_origin_npy=test_visits_origin_npy,
        visits_274_npy=test_visits_274_npy,
        visits_224_npy=test_visits_224_npy
    )
    train_visits = load_visits(**train_args)
    test_visits = load_visits(**test_args)
    print(train_visits.shape)
    for i in range(3):
        print(train_visits[i].shape)
        print(test_visits[i].shape)
    # print(visits[28888])
    # print(visits[28888][4][9][3])


if __name__ == '__main__':
    main()
