import random
def make_burst_mask(n_slots, duty, min_len, rng):
        """
        生成长度为 n_slots 的 0/1 mask：
        - 1 表示该 5Hz slot 打刺激
        - 0 表示该 5Hz slot 无刺激
        mask 的 1 会以 burst(连续1) 的形式出现，burst 起点随机
        """
        if n_slots <= 0:
            return []
        # 0 ≤ n_on ≤ n_slots 且为有效整数
        n_on = int(round(n_slots * duty))
        n_on = max(0, min(n_on, n_slots))
        n_off = n_slots - n_on

        # 极端情况：不刺激 or 全刺激
        if n_on == 0:
            return [0] * n_slots
        if n_on == n_slots:
            return [1] * n_slots

        # 如果要求 burst 最小长度，但 n_on < min_len，无法满足
        # 这里选择：降级成一个 burst（长度=n_on）
        if n_on < min_len:
            return [1] * n_on + [0] * (n_slots - n_on)

        # burst 数量的上限，刺激被分为几组 burst：
        # 1) 每个 burst 至少 min_len 个 1 => b <= floor(n_on / min_len)
        # 2) burst 之间至少留 1 个 0（否则会合并为更长 burst）
        #    n_off 个 0 最多能分隔成 b-1 个内部间隔 => b <= n_off + 1
        max_b = min(n_on // min_len, n_off + 1)
        b = rng.randint(1, max_b)

        # 先给每个 burst 分配 min_len 个 1
        burst_lens = [min_len] * b
        remaining = n_on - min_len * b

        # 把剩余的 1 随机分配到各个 burst 上（让 burst 长度有随机性）
        for _ in range(remaining):
            burst_lens[rng.randrange(b)] += 1

        # 现在分配 0：总共有 n_off 个 0
        # 内部 gap（burst 之间）至少 1 个 0，共 (b-1) 个内部 gap
        internal_min = b - 1
        # 理论上 b <= n_off + 1 已保证 n_off >= internal_min
        zeros_left = n_off - internal_min

        # gap 一共有 b+1 段：前导、(b-1)个内部、尾随
        gaps = [0] * (b + 1)
        # 先给每个内部 gap 1 个 0
        for gi in range(1, b):
            gaps[gi] = 1

        # 剩余 0 随机撒到所有 gap（包括前导/尾随/内部）
        for _ in range(zeros_left):
            gaps[rng.randrange(b + 1)] += 1

        # 拼接 mask： 0... + 111.. + 0.. + 111.. + ... + 0...
        mask = []
        mask.extend([0] * gaps[0])
        for idx, L in enumerate(burst_lens):
            mask.extend([1] * L)
            mask.extend([0] * gaps[idx + 1])

        # 保险：裁剪到 n_slots（理论上应正好等长）
        return mask[:n_slots]

if __name__ == '__main__' :
    print(make_burst_mask(20,1,3,rng = random.Random(None)))