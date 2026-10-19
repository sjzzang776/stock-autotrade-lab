"""비중 계산 도우미 — 모의투자(paper)와 백테스트가 함께 씀 (공개 저장소에 매매 코드 없이 백테스트만 올리려고 분리, 2026-10-19)"""
import numpy as np


def capped_weights(w, cap):
    """비례 비중(합 1)을 종목당 cap 으로 자르고 남는 몫을 상한 아래 종목에 비례 재분배"""
    w = np.asarray(w, float)
    w = w / w.sum()
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        extra = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w > 0)
        if not free.any():
            break
        w[free] += extra * w[free] / w[free].sum()
    return w
