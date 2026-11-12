"""试采读段规划。

约束：
- 每个测点占用保持寄存器上连续 ``words`` 个地址（``words`` 为 1 或 2），
  且必须在一次 Modbus 读取中整段读完；
- 每次读取一段连续寄存器，读段整体必须落在设备可读区间
  ``[range_start, range_end]``（含两端）内，读取字数不超过 ``max_read``。

优化目标（字典序）：
1. 请求次数最少；
2. 请求次数相同时，读取寄存器总数最少。

思路：同址同字数的测点对规划完全等价（读段是否覆盖某地址区间与重复条数
无关），因此先按 ``(address, words)`` 去重，只在去重后的测点上按起始地址
排序做动态规划。``dp[i]`` 表示覆盖第 i 个及之后所有去重测点的最优方案
``(请求次数, 寄存器总数)``，枚举第一个读段覆盖的测点区间 ``i..j`` 进行转移。
读段从第 i 个测点的起始地址开始，结束于区间内测点的最大末端，因此读段紧贴
测点，不会多读无用寄存器。重建读段时再把每组测点的全部原下标展开回去。

复杂度：设 n 为提交条数、m 为去重后测点数（m <= n），去重 O(n)，
动态规划 O(m^2) 时间、O(m) 空间——规划开销不随同一测点的重复条数增长。
"""

from __future__ import annotations

from dataclasses import dataclass


class PlanError(ValueError):
    """无可行方案。"""


@dataclass(frozen=True)
class Point:
    address: int
    words: int

    @property
    def end(self) -> int:
        """测点占用的最后一个寄存器地址（含）。"""
        return self.address + self.words - 1


@dataclass(frozen=True)
class Segment:
    """一次 Modbus 读取对应的连续寄存器段。"""

    start: int
    end: int
    point_indexes: tuple[int, ...]

    @property
    def count(self) -> int:
        return self.end - self.start + 1


def plan_reads(
    points: list[tuple[int, int]],
    range_start: int,
    range_end: int,
    max_read: int,
) -> list[Segment]:
    """计算最优读段方案。

    :param points: ``(起始地址, 字数)`` 列表，下标即请求中的测点序号；
        同一 ``(起始地址, 字数)`` 可重复出现，规划时只算一条，
        生成读段时会把重复条目的原下标全部列回
    :raises PlanError: 不存在可行方案时抛出，消息为中文原因
    """
    if range_start < 0 or range_end < range_start:
        raise PlanError(f"可读区间非法：[{range_start}, {range_end}]")
    if max_read < 1:
        raise PlanError("单次读取上限必须 >= 1")

    pts = [Point(address=int(addr), words=int(words))
           for addr, words in points]

    # 单点可行性校验：每个测点自身必须能被一次读取覆盖
    for p in pts:
        if p.address < range_start or p.end > range_end:
            raise PlanError(
                f"测点 address={p.address} words={p.words} 超出可读区间 "
                f"[{range_start}, {range_end}]（含两端），无可行方案"
            )
        if p.words > max_read:
            raise PlanError(
                f"测点 address={p.address} words={p.words} 超过单次读取上限 "
                f"{max_read}，无法一次读完"
            )

    # 同址同字数去重：key=(address, words)，value=该测点对应的全部原下标。
    # 规划只与去重后的测点有关，重复条数再大也不进入动态规划。
    group_indexes: dict[tuple[int, int], list[int]] = {}
    for idx, p in enumerate(pts):
        group_indexes.setdefault((p.address, p.words), []).append(idx)

    uniq = sorted(
        (Point(address=addr, words=words) for addr, words in group_indexes),
        key=lambda p: (p.address, p.words),
    )
    m = len(uniq)

    # dp[i] = 覆盖排序后第 i..m-1 个去重测点的最优 (请求次数, 寄存器总数)
    # choice[i] = 最优方案中第一个读段覆盖到的去重测点下标 j（读段含 i..j）
    inf: tuple[float, float] = (float("inf"), float("inf"))
    dp: list[tuple[float, float]] = [inf] * (m + 1)
    choice: list[int] = [-1] * m
    dp[m] = (0, 0)

    for i in range(m - 1, -1, -1):
        seg_start = uniq[i].address
        seg_end = uniq[i].end
        best: tuple[float, float] = inf
        best_j = -1
        for j in range(i, m):
            if uniq[j].end > seg_end:
                seg_end = uniq[j].end
            count = seg_end - seg_start + 1
            # 地址单调，seg_end 不减，超限后 j 再增大也不可能可行
            if count > max_read or seg_end > range_end:
                break
            candidate = (1 + dp[j + 1][0], count + dp[j + 1][1])
            # 严格小于才替换：次数与寄存器数都相同时保留最早切点，结果确定
            if candidate < best:
                best = candidate
                best_j = j
        if best_j < 0:
            raise PlanError("测点无法在单次读取上限内组合为连续读段，无可行方案")
        dp[i] = best
        choice[i] = best_j

    # 重建读段：按去重测点的 (address, words) 顺序展开，每组原下标升序，
    # 与去重前按 (address, words, index) 排序得到的段内下标顺序一致。
    segments: list[Segment] = []
    i = 0
    while i < m:
        j = choice[i]
        seg_start = uniq[i].address
        seg_end = max(p.end for p in uniq[i:j + 1])
        indexes: list[int] = []
        for p in uniq[i:j + 1]:
            indexes.extend(group_indexes[(p.address, p.words)])
        segments.append(
            Segment(start=seg_start, end=seg_end, point_indexes=tuple(indexes))
        )
        i = j + 1
    return segments
