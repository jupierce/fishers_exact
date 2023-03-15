# Using fast-fisher: https://github.com/MrTomRod/fast-fisher
import bitstring
import math
import os
import fast_fisher.fast_fisher_cython
import scipy
from multiprocessing import Pool

import time

# Set to true to sanity check the generated table against scipy calculated values
scipy_sanity_check = True

MAX_CELL_SAMPLES = 32   # must be a power of 2 and divisible by 8
# MAX_CELL_SAMPLES = 32
# MAX_CELL_SAMPLES = 8
MULTIPLICATION_SHIFTS = int(math.log2(MAX_CELL_SAMPLES))

if (1 << MULTIPLICATION_SHIFTS) != MAX_CELL_SAMPLES or MAX_CELL_SAMPLES//8*8 != MAX_CELL_SAMPLES:
    print(f'MAX_CELL_SAMPLES must be a power of 2 and divisible by 8')
    exit(1)

ALPHA = 0.05


def offset(a, b, c, d) -> int:
    return (a << (MULTIPLICATION_SHIFTS * 3)) + (b << (MULTIPLICATION_SHIFTS * 2)) + (c << (MULTIPLICATION_SHIFTS * 1)) + d


def fisher_for_a(a: int):
    result = bitstring.BitArray(length=MAX_CELL_SAMPLES * MAX_CELL_SAMPLES * MAX_CELL_SAMPLES)
    for b in range(0, MAX_CELL_SAMPLES):
        for c in range(0, MAX_CELL_SAMPLES):
            for d in range(0, MAX_CELL_SAMPLES):
                pvalue = fast_fisher.fast_fisher_cython.fisher_exact(a, b, c, d, alternative='greater')
                if pvalue <= ALPHA:
                    result.set(True, offset(0, b, c, d))
    return a, result


if __name__ == '__main__':
    start = time.time_ns()

    if MAX_CELL_SAMPLES > 64 and scipy_sanity_check:
        print('It will take too long to sanity check computed tables with > 64 samples')
        exit(1)

    OUTPUT_FILENAME = 'table-gen.bin'

    with open(OUTPUT_FILENAME, 'w+b') as f:
        with Pool(max(4, os.cpu_count() - 2)) as p:
            for result in p.imap_unordered(fisher_for_a, range(0, MAX_CELL_SAMPLES)):
                a, bits = result
                f.seek(a * (bits.length // 8))
                bits.tofile(f)
                print(result)

    if scipy_sanity_check:
        with open(OUTPUT_FILENAME, 'rb') as f:
            all_bits = bitstring.BitArray(f)
            for a in range(0, MAX_CELL_SAMPLES):
                print(f'Sanity checking: {a} of {MAX_CELL_SAMPLES-1}')
                for b in range(0, MAX_CELL_SAMPLES):
                    for c in range(0, MAX_CELL_SAMPLES):
                        for d in range(0, MAX_CELL_SAMPLES):
                            bit_offset = offset(a, b, c, d)
                            _, pvalue = scipy.stats.fisher_exact([[a, b], [c, d]], alternative='greater')
                            significant = (pvalue <= ALPHA)
                            if significant != all_bits[bit_offset]:
                                fisher_fast_pvalue = fast_fisher.fast_fisher_cython.fisher_exact(a, b, c, d, alternative="greater")
                                if abs(fisher_fast_pvalue - pvalue) > 0.0000001:
                                    print(f'Difference at a={a} b={b} c={c} d={d} ; scipy got={pvalue}    and   fast got={fisher_fast_pvalue}')
                            else:
                                byte_offset = bit_offset // 8
                                bit_shift = 7 - (bit_offset % 8)
                                f.seek(byte_offset)
                                v = f.read(1)[0]
                                file_based_significant = (v & (1 << bit_shift)) > 0
                                if significant != file_based_significant:
                                    print(f'Difference at a={a} b={b} c={c} d={d} in file based read; scipy got={significant}    and   file got bit_offset={bit_offset}  byte_offset={byte_offset}  bit_shift={bit_shift} = {v}')
