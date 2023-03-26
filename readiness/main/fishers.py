import fast_fisher.fast_fisher_cython
from math import log2
import mmap
import traceback

from readiness.settings import BASE_DIR

MAX_CELL_SAMPLES = 512   # must be a power of 2 and divisible by 8
MULTIPLICATION_SHIFTS = int(log2(MAX_CELL_SAMPLES))

if (1 << MULTIPLICATION_SHIFTS) != MAX_CELL_SAMPLES or MAX_CELL_SAMPLES//8*8 != MAX_CELL_SAMPLES:
    print(f'MAX_CELL_SAMPLES must be a power of 2 and divisible by 8')
    exit(1)

ALPHA = 0.05

try:
    table_bin = open(f'{BASE_DIR}/../table.bin', 'rb')
    table_mmap = mmap.mmap(table_bin.fileno(), length=0, access=mmap.ACCESS_READ)
except:
    print(f'WARNING: Precomputed fishers will not be used.')
    table_mmap = None
    traceback.print_exc()


def fisher_offset(a, b, c, d) -> int:
    # TODO: use chi^2 instead of normalizing to 500 samples?
    if a > 500 or b > 500:
        reducer = min((500 / max(a, 1)), (500 / max(b, 1)))  # use max in case one of the values is 0
        a = int(a * reducer)
        b = int(b * reducer)
    if c > 500 or d > 500:
        reducer = min((500 / max(c, 1)), (500 / max(d, 1)))
        c = int(c * reducer)
        d = int(d * reducer)
    return (a << (MULTIPLICATION_SHIFTS * 3)) + (b << (MULTIPLICATION_SHIFTS * 2)) + (c << (MULTIPLICATION_SHIFTS * 1)) + d


def fisher_significant(a, b, c, d, alpha=ALPHA) -> bool:
    if (not table_mmap) or a > 500 or b > 500 or c > 500 or d > 500 or alpha != ALPHA:
        return fast_fisher.fast_fisher_cython.fisher_exact(a, b, c, d, alternative='greater') < alpha
    # Otherwise, look it up faster in the precomputed data
    offset = fisher_offset(a, b, c, d)
    bin_offset = offset // 8
    bit_shift = 7 - (offset % 8)
    return table_mmap[bin_offset] & (1 << bit_shift) > 0
