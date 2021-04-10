# *_*coding:utf-8 *_*
"""
这个脚本是用来压缩数据，减少内存占用的。压缩数据有几种方法：

第一种：
当我们明确知道要加载数据的范围，使用pd.read_table读取数据时，可以用其中的dtype参数来手动指定类型。比如某一列的数据范围肯定在0~255之中，那么我们可以指定为np.uint8类型。

第二种：
如果数据列数太多，或者不清楚数据具体范围的话下面是一个脚本，可以自动判断类型，并根据类型修改数据范围。

第三种：
批量处理，增量训练模型。
"""
import numpy as np

def reduce_mem_usage(props):
    # 计算当前内存
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of the dataframe is :", start_mem_usg, "MB")
    # 哪些列包含空值，空值用-999填充。why：因为np.nan当做float处理
    NAlist = []
    for col in props.columns:
        # 这里只过滤了objectd格式，如果你的代码中还包含其他类型，请一并过滤
        if (props[col].dtypes != object):
            # 判断是否是int类型
            isInt = False
            mmax = props[col].max()
            mmin = props[col].min()
            # Integer 不支持 NA,所以要填充
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(-999, inplace=True)  # 用-999填充
            # 试试看能不能转int
            asint = props[col].fillna(0).astype(np.int64)
            result = np.fabs(props[col] - asint)
            result = result.sum()
            if result < 0.01:  # 绝对误差和小于0.01认为可以转换的，要根据task修改
                isInt = True
            # 生成整数/无符号的int类型
            if isInt:
                if mmin >= 0:  # 最小值大于0，转换成无符号整型
                    if mmax <= 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mmax <= 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mmax <= 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:  # 转换成有符号整型
                    if mmin > np.iinfo(np.int8).min and mmax < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mmin > np.iinfo(np.int16).min and mmax < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mmin > np.iinfo(np.int32).min and mmax < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mmin > np.iinfo(np.int64).min and mmax < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)
            # 注意：这里对于float都转换成float16，需要根据自己的情况自己更改
            else:
                props[col] = props[col].astype(np.float16)

    mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props, NAlist
