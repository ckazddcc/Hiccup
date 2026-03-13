import numpy as np
from itertools import combinations
import yaml


def generate_n_compositions(elements_boundary, total_atoms=48, min_others=9):
    """
    通用生成 N 元组分函数
    :param n_elements: 元素种类数 (N)
    :param total_atoms: 原子总数 (S)
    :param min_others: 约束条件：除了当前元素外，其他元素总数至少要有几个
                        (对应你原代码中的 total - 8 限制)
    """

    # 使用“隔板法”思路生成所有和为 total_atoms 的非负整数序列
    # 这里的 logic 是：在 (total + n - 1) 个位置中选 (n - 1) 个位置放隔板
    def compositions_generator(n, total):
        if n == 1:
            yield (total,)
            return
        for i in range(total + 1):
            for rest in compositions_generator(n - 1, total - i):
                yield (i,) + rest

    all_comps = []
    for comp in compositions_generator(len(elements_boundary.keys()), total_atoms):
        # 应用约束：任何一个元素的数量都不能超过 max_val
        keys = elements_boundary.keys()
        if all(elements_boundary[key][0] <= x <= elements_boundary[key][1] for key, x in zip(keys, comp)):
            all_comps.append(comp)

    return np.array(all_comps)


def farthest_point_sampling(points, k):
    """
    通用 FPS 采样 (支持 N 维)
    """
    if len(points) <= k:
        return np.arange(len(points))

    selected = []
    # 1. 随机选一个起点
    idx = np.random.randint(len(points))
    selected.append(idx)

    # 初始化距离数组为无穷大
    distances = np.full(len(points), np.inf)

    for _ in range(k - 1):
        last_point = points[selected[-1]]
        # 计算当前点到所有点的欧氏距离 (N维)
        dist = np.linalg.norm(points - last_point, axis=1)
        # 更新每个点到已选集合的最小距离
        distances = np.minimum(distances, dist)
        # 选取距离已选集合最远的点
        next_idx = np.argmax(distances)
        selected.append(next_idx)

    return np.array(selected)


def sample_compositions_n(elements_boundary, total_atoms, substrate, n_samples):
    # Step 1: 生成 N 维空间下的所有合法组分
    all_comps = generate_n_compositions(elements_boundary, total_atoms)

    # Step 2: 归一化到比例空间 (0~1 之间)，方便 FPS 计算
    ratios = all_comps / total_atoms

    # Step 3: 执行 FPS 采样
    idx = farthest_point_sampling(ratios, n_samples)

    update_compos = []
    for compos in all_comps[idx]:
        _compos = list(compos)
        for i, elem in enumerate(elements_boundary.keys()):
            _compos[i] += substrate.get(elem, 0)
            update_compos.append(_compos)
            # print(compos, _compos)

    return update_compos

def sample_composition(config_path):
    config = yaml.safe_load(open(config_path))
    result = sample_compositions_n(elements_boundary=config["elements_boundary"],
                                   total_atoms=config["total_atoms"],
                                   substrate=config["substrate"],
                                   n_samples=config["n_samples"])
    print("All compositions:")
    print(result)


if __name__ == '__main__':
    sample_composition(config_path='/home/cchen/Hiccup/template/fps_config.yml')



