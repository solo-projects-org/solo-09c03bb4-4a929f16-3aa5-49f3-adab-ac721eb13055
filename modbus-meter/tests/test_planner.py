"""读段规划单元测试（不需要设备与网络）。"""

from __future__ import annotations

import time

import pytest

from app.planner import PlanError, plan_reads


def seg_tuples(segments):
    return [(s.start, s.end, tuple(s.point_indexes)) for s in segments]


def test_single_word_points_are_merged_when_within_limit():
    # 可读 0..99，上限 16；三个相邻单字应合并为 1 段、共 3 个寄存器
    segs = plan_reads(
        points=[(0, 1), (1, 1), (2, 1)],
        range_start=0,
        range_end=99,
        max_read=16,
    )
    assert seg_tuples(segs) == [(0, 2, (0, 1, 2))]
    assert sum(s.count for s in segs) == 3


def test_double_word_point_and_neighbors():
    # 测点 0(双字) 占 0..1；测点 2(单字)；上限 2 -> 必须两段
    segs = plan_reads(
        points=[(0, 2), (2, 1)],
        range_start=0,
        range_end=99,
        max_read=2,
    )
    assert seg_tuples(segs) == [(0, 1, (0,)), (2, 2, (1,))]
    assert [s.count for s in segs] == [2, 1]


def test_dp_beats_greedy_on_register_totals():
    # 贪心（尽量多吞测点）：第一段 0..5 含点0、点1（6 个寄存器），
    #   并入点2(双字占6..7)后为 0..7 共 8 字 > 上限 6，被迫断开；
    #   第二段 6..8 含点2、点3（3 个寄存器）-> 共 9。
    # DP 找到更优：[0..0](1, 点0) + [5..8](4, 点1..3) = 2 次共 5。
    segs = plan_reads(
        points=[(0, 1), (5, 1), (6, 2), (8, 1)],
        range_start=0,
        range_end=99,
        max_read=6,
    )
    assert len(segs) == 2
    assert seg_tuples(segs) == [(0, 0, (0,)), (5, 8, (1, 2, 3))]
    assert sum(s.count for s in segs) == 5


def test_dp_chooses_smaller_register_total_when_requests_tie():
    # 真正的反贪心场景：
    # 点 0(单字)、点 6(单字)、点 7(双字占 7..8)；上限 6，可读 0..99。
    # 贪心：[0..5](6, 含点0) + [6..8](3, 含点1、点2) = 2 次共 9。
    # DP  ：[0..0](1, 含点0) + [6..8](3, 含点1、点2) = 2 次共 4，更优。
    segs = plan_reads(
        points=[(0, 1), (6, 1), (7, 2)],
        range_start=0,
        range_end=99,
        max_read=6,
    )
    assert len(segs) == 2
    assert seg_tuples(segs) == [(0, 0, (0,)), (6, 8, (1, 2))]
    assert sum(s.count for s in segs) == 4


def test_point_outside_readable_range_rejected():
    with pytest.raises(PlanError, match="超出可读区间"):
        plan_reads(points=[(50, 1)], range_start=0, range_end=49, max_read=16)


def test_double_word_crossing_range_end_rejected():
    # 起始 49、双字占 49..50，区间止于 49
    with pytest.raises(PlanError, match="超出可读区间"):
        plan_reads(points=[(49, 2)], range_start=0, range_end=49, max_read=16)


def test_words_exceeding_max_read_rejected():
    with pytest.raises(PlanError, match="超过单次读取上限"):
        plan_reads(points=[(0, 2)], range_start=0, range_end=99, max_read=1)


def test_duplicate_points_are_covered_together():
    segs = plan_reads(
        points=[(3, 1), (3, 1), (4, 1)],
        range_start=0,
        range_end=99,
        max_read=16,
    )
    assert seg_tuples(segs) == [(3, 4, (0, 1, 2))]


def test_invalid_range_rejected():
    with pytest.raises(PlanError):
        plan_reads(points=[(0, 1)], range_start=10, range_end=5, max_read=16)


def test_unsorted_input_returns_original_indexes():
    segs = plan_reads(
        points=[(20, 1), (0, 1), (10, 2)],
        range_start=0,
        range_end=99,
        max_read=32,
    )
    # 全部并入一段（20..0 含 21 个寄存器，<= 上限 32），point_indexes 保留请求下标
    assert seg_tuples(segs) == [(0, 20, (1, 2, 0))]


def test_duplicate_points_list_every_original_index():
    # 同址同字数重复 3 条，再带一个相邻测点；规划只看去重测点，
    # 但读段必须列全所有原下标
    segs = plan_reads(
        points=[(3, 1), (3, 1), (4, 1), (3, 1)],
        range_start=0,
        range_end=99,
        max_read=16,
    )
    assert seg_tuples(segs) == [(3, 4, (0, 1, 3, 2))]


def test_same_address_different_words_form_distinct_groups():
    # 地址 10：两条单字（下标 0、1）+ 一条双字占 10..11（下标 2），
    # 再加地址 11 的单字（下标 3）。去重后 3 个测点，全部并入 10..11 一段。
    segs = plan_reads(
        points=[(10, 1), (10, 1), (10, 2), (11, 1)],
        range_start=0,
        range_end=99,
        max_read=8,
    )
    assert seg_tuples(segs) == [(10, 11, (0, 1, 2, 3))]

    # 上限 2 时：10..11 一段即可同时完整覆盖双字（10..11）与 11 的单字，
    # 仍是一段；段内按 (address, words) 排序后展开原下标
    segs = plan_reads(
        points=[(10, 1), (10, 1), (10, 2), (11, 1)],
        range_start=0,
        range_end=99,
        max_read=2,
    )
    assert seg_tuples(segs) == [(10, 11, (0, 1, 2, 3))]


def test_duplicate_rows_do_not_inflate_plan_cost():
    # 同一批去重测点：重复 1 条与重复 200 条，方案（读段与计数）必须完全一致
    base = [(0, 1), (5, 1), (6, 2), (8, 1)]
    once = plan_reads(points=base, range_start=0, range_end=99, max_read=6)
    many = plan_reads(
        points=base * 200,
        range_start=0,
        range_end=99,
        max_read=6,
    )
    # 读段起止与字数一致
    assert [(s.start, s.end, s.count) for s in many] == [
        (s.start, s.end, s.count) for s in once
    ]
    # 每组原下标都按重复次数展开，且总数等于提交条数
    assert sum(len(s.point_indexes) for s in many) == len(base) * 200
    for seg in many:
        assert all(0 <= idx < len(base) * 200 for idx in seg.point_indexes)
    # 最优性不变：2 次请求、共 5 个寄存器
    assert len(many) == 2
    assert sum(s.count for s in many) == 5


def test_planning_time_does_not_grow_quadratically_with_duplicate_rows():
    # 去重后测点数固定为 200 个稀疏单字，重复条数从 10 提到 10000；
    # 去重后规划开销与重复条数无关，耗时不应随条数平方放大。
    distinct = [(a * 10, 1) for a in range(200)]

    def elapsed(rows: int) -> float:
        repeats, tail = divmod(rows, len(distinct))
        payload = distinct * repeats + distinct[:tail]
        start = time.perf_counter()
        plan_reads(points=payload, range_start=0, range_end=100000, max_read=4)
        return time.perf_counter() - start

    small = elapsed(200)
    huge = elapsed(10000)
    # 宽松上限防抖动；旧的 O(n^2) 实现在该数据量下需数秒，必然超界
    assert huge < 1.0
    assert huge < max(small * 10, 0.2)


def test_massive_duplicates_at_one_address_remain_fast():
    # 最极端场景：同一单字测点重复 20000 条，去重后 m=1，规划应近乎瞬时
    start = time.perf_counter()
    segs = plan_reads(
        points=[(42, 1)] * 20000,
        range_start=0,
        range_end=99,
        max_read=16,
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
    assert len(segs) == 1
    assert segs[0].point_indexes == tuple(range(20000))
