import pandas as pd
import numpy as np
import os

s_column_name = {
    'sign': 1,
    'shot_line': 10,
    'shot_point': 10,
    'index': 3,
    'code': 2,
    'static': 4,
    'deep': 4,
    'base_level': 4,
    'tb': 2,
    'water_deep': 6,
    'X': 9,
    'Y': 10,
    'elevation': 6,
    'day': 3,
    'time': 6
}

r_column_name = {
    'sign': 1,
    'rec_line': 10,
    'rec_point': 10,
    'index': 3,
    'code': 2,
    'static': 4,
    'deep': 4,
    'base_level': 4,
    'tb': 2,
    'water_deep': 6,
    'X': 9,
    'Y': 10,
    'elevation': 6,
    'day': 3,
    'time': 6
}

x_column_name = {
    'sign': 1,
    'tape_num': 6,
    'ff_id': 8,
    'record_increment': 1,
    'equipment_code': 1,
    'shot_line': 10,
    'shot_point': 10,
    'index': 1,
    'start_channel': 5,
    'end_channel': 5,
    'channel_increment': 1,
    'receiver_line': 10,
    'start_receiver_point': 10,
    'end_receiver_point': 10,
    'receiver_index': 1
}


def sps_read(mode, s_file_path, r_file_path, x_file_path):
    if mode == 'sx':
        return read_shot(s_file_path), read_x(x_file_path)
    if mode == 'all':
        return read_shot(s_file_path), read_rec(r_file_path), read_x(x_file_path)


def read_shot(s_file_path):
    s_file = pd.read_csv(s_file_path, header=None)
    s_head_num = 0
    step = 0
    for i in range(s_file.shape[0]):
        if s_file.loc[i][0][0] == 'H':
            s_head_num += 1
        else:
            break

    shot_head = s_file[0:s_head_num]
    s_file = s_file.drop(s_file.index[0:s_head_num])
    df_s = pd.DataFrame(columns=s_column_name.keys())
    for col in s_column_name:
        df_s[col] = s_file[0].str.slice(start=step, stop=step + s_column_name[col])
        step += s_column_name[col]
    return df_s, shot_head


def read_rec(r_file_path):
    # 头文件含多个分隔符要使用sep=None，不含多个则用默认参数，使用了sep=None会导致读取内容出错
    r_file = pd.read_csv(r_file_path, header=None, sep=None)
    r_head_num = 0
    for i in range(r_file.shape[0]):
        if r_file.loc[i][0][0] == 'H':
            r_head_num += 1
        else:
            break

    rec_head = r_file[0:r_head_num]
    df_r = pd.DataFrame(columns=r_column_name.keys())
    r_file = r_file.drop(r_file.index[0:r_head_num])
    step = 0
    for col in r_column_name:
        df_r[col] = r_file[0].str.slice(start=step, stop=step + r_column_name[col])
        step += r_column_name[col]
    return df_r, rec_head


def read_x(x_file_path):
    x_file = pd.read_csv(x_file_path, header=None)
    x_head_num = 0
    for i in range(x_file.shape[0]):
        if x_file.loc[i][0][0] == 'H':
            x_head_num += 1
        else:
            break

    relation_head = x_file[0:x_head_num]
    x_file = x_file.drop(x_file.index[0:x_head_num])
    df_x = pd.DataFrame(columns=x_column_name.keys())
    step = 0
    for col in x_column_name:
        df_x[col] = x_file[0].str.slice(start=step, stop=step + x_column_name[col])
        step += x_column_name[col]
    return df_x, relation_head


def read_ob(ob_file_path):
    pass


# 头文件输出待后续开发
def sps_out(df, out_name):
    df = df.astype(str)
    if out_name[-1] == 's' or out_name[-1] == 'S':
        for column in df:
            # 按照s_column_name字典中给定的长度输出右对齐格式
            df[column] = df[column].map(lambda a: format(a, '>{}'.format(s_column_name[column])))
    if out_name[-1] == 'r' or out_name[-1] == 'R':
        for column in df:
            # 按照s_column_name字典中给定的长度输出右对齐格式
            df[column] = df[column].map(lambda a: format(a, '>{}'.format(r_column_name[column])))
    if out_name[-1] == 'x' or out_name[-1] == 'X':
        for column in df:
            # 按照s_column_name字典中给定的长度输出右对齐格式
            df[column] = df[column].map(lambda a: format(a, '>{}'.format(x_column_name[column])))
    df_out = df.iloc[:, 0].str.cat(others=[df.iloc[:, i] for i in range(1, len(df.columns))], sep=None)
    # if headers:
    #     head = read_shot()
    #     df_out = pd.concat([head, df_out])
    df_out.to_csv(out_name, header=False, index=False)


def create_shot_file(shot_coordinate_file):
    # 读坐标
    shot_coordinate_excel = pd.read_csv(shot_coordinate_file, header=None)
    _shot_file = pd.DataFrame(columns=s_column_name.keys())
    _shot_file['shot_line'] = shot_coordinate_excel[0].str.split('-', expand=True)[0]
    if (_shot_file['shot_line'].str.len() > 10).any():
        _shot_file['shot_line'][_shot_file['shot_line'].str.len() > 3] = \
            _shot_file['shot_line'].str.split('.', expand=True)[0] + ['2.'] * _shot_file.shape[0] + \
            _shot_file['shot_line'].str.split('.', expand=True)[1]
        _shot_file['shot_line'][_shot_file['shot_line'].str.len() <= 3] = \
            _shot_file['shot_line'] + ['2'] * _shot_file.shape[0]
    else:
        _shot_file['shot_line'] = _shot_file['shot_line'] + ['2'] * _shot_file.shape[0]
    _shot_file['shot_point'] = shot_coordinate_excel[0].str.split('-', expand=True)[1]
    _shot_file['X'] = shot_coordinate_excel[2].round(1)
    _shot_file['Y'] = shot_coordinate_excel[1].round(1)
    # 默认赋值
    _shot_file['sign'] = 'S'
    _shot_file['index'] = 1
    _shot_file['code'] = 'E1'
    _shot_file = _shot_file.fillna(' ')
    return _shot_file


def create_receiver_file(receiver_coordinate_file):
    # 读坐标
    rec_coordinate_excel = pd.read_csv(receiver_coordinate_file, header=None)
    _rec_file = pd.DataFrame(columns=r_column_name.keys())
    _rec_file['rec_line'] = rec_coordinate_excel[0].str.split('-', expand=True)[0] + \
                            rec_coordinate_excel[0].str.split('-', expand=True)[1]
    _rec_file['rec_point'] = rec_coordinate_excel[0].str.split('-', expand=True)[2]
    _rec_file['X'] = rec_coordinate_excel[2].round(1)
    _rec_file['Y'] = rec_coordinate_excel[1].round(1)
    # 默认赋值
    _rec_file['sign'] = 'R'
    _rec_file['index'] = 1
    _rec_file['code'] = 'G1'
    _rec_file = _rec_file.fillna(' ')
    return _rec_file


def create_relation_file(_shot_file, channel_increment=480, line_num=1):
    # 计算起始通道号
    # channel_start = np.arange(1, channel_increment * line_num, channel_increment)
    channel_start = [1, 541, 1621]
    # 计算终止通道号
    # channel_end = np.arange(channel_increment, channel_increment * (line_num + 1), channel_increment)
    channel_end = [540, 1620, 2160]
    _relation_file = pd.DataFrame(columns=x_column_name.keys())
    _relation_file['shot_line'] = np.repeat(_shot_file['shot_line'].values, line_num, axis=0)
    _relation_file['shot_line'] = _relation_file['shot_line'].astype('str')
    _relation_file['shot_point'] = np.repeat(_shot_file['shot_point'].values, line_num, axis=0)
    _relation_file['shot_point'] = _relation_file['shot_point'].astype('str')
    if (_relation_file['shot_line'].str.len() > 10).any():
        _relation_file['receiver_line'][_relation_file['shot_line'].str.len() > 4] = \
            _relation_file['shot_line'].str.split('.', expand=True)[0].str.slice(0, -1) + ['1', '2', '3'] * \
            _shot_file.shape[0] + '.' + _relation_file['shot_line'].str.split('.', expand=True)[1]
        _relation_file['receiver_line'][_relation_file['shot_line'].str.len() <= 4] = \
            _relation_file['shot_line'].str.split(3, expand=True)[0] + ['1', '2', '3'] * _shot_file.shape[0]
    else:
        _relation_file['receiver_line'] = _relation_file['shot_line'].str.slice(0, -1) + ['1', '2', '3'] * \
                                          _shot_file.shape[0]
    _relation_file['start_channel'] = list(channel_start) * _shot_file.shape[0]
    _relation_file['end_channel'] = list(channel_end) * _shot_file.shape[0]
    _relation_file['start_receiver_point'] = _relation_file['shot_point'].astype(float).astype(int) - channel_increment / 2 + 1
    _relation_file['end_receiver_point'] = _relation_file['shot_point'].astype(float).astype(int) + channel_increment / 2
    # 默认赋值
    _relation_file['sign'] = 'X'
    _relation_file['tape_num'] = 1
    _relation_file['record_increment'] = 1
    _relation_file['equipment_code'] = 1
    _relation_file['index'] = 1
    _relation_file['channel_increment'] = 1
    _relation_file['receiver_index'] = 1
    _relation_file = _relation_file.fillna(' ')
    return _relation_file


# 整理线号的代码拆分出来，简化sps生成函数
def sort_line_nbr():
    pass


file_path = '../../OneDrive - stu.cdut.edu.cn/南江北投标/宁强、南江北炮检点坐标/'
shot_file_name = '南江炮线非攻关.csv'
# shot_file_name = '宁强炮点.csv'
rec_file_name = '南江检波线非攻关.csv'
# rec_file_name = '宁强检波点.csv'

shot_file = create_shot_file(file_path + shot_file_name)
rec_file = create_receiver_file(file_path + rec_file_name)
# if (shot_file['shot_line'] + shot_file['shot_line']).duplicated().bool():
#     print('炮点有重复')
relation_file = create_relation_file(shot_file, channel_increment=480, line_num=3)

if __name__ == '__main__':
    file_out_name = '南江北'
    sps_out(shot_file, './{}.S'.format(file_out_name))
    sps_out(rec_file, './{}.R'.format(file_out_name))
    sps_out(relation_file, './{}.X'.format(file_out_name))
    # (s, s_head), (x, x_head) = sps_read('sx', './s.S', './r.R', './x.X')
    # sps_out(s, './nq.S')
    # 测试更新
