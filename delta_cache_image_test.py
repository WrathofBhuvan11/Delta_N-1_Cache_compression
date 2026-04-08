#!/usr/bin/env python3
"""
Three compression approaches compared on real image data:

  1. XOR CACHE    -- XOR the target with the single best partner line (N=1).
                    Simple, symmetric, no carry. The baseline from the XOR Cache paper.
                    Uses an unsigned bit-width cost metric (see entropy_byte_xor).
  2. BYTE GREEDY  -- Find the globally best single arithmetic base, then greedily
                    add further bases if each reduces cost. Adaptive N, unordered.
  3. SAD DELTA    -- Use Sum of Absolute Differences (hardware-cheap proxy)
                    to pick the best single base. Near-optimal compression
                    with fewer ops than greedy (see measured timing below).
# ipython delta_cache_image_test.py lenna_test.png
"""

import math
import sys
import os
import time
from PIL import Image


# ---- how many bits does a subtraction delta byte need? ----
# treat it as signed (-128 to 127), count the bits needed to store the magnitude
# zero is free (0 bits)
def entropy_byte(val):
    v = val & 0xFF
    signed_v = v if v < 128 else v - 256  # wrap around if > 127
    mag = abs(signed_v)
    return 0 if mag == 0 else math.ceil(math.log2(mag + 1))


# Entropy for XOR results - XOR bytes are unsigned so on't sign-extend
# e.g. 0x03 XOR 0x05 = 0x06, that's just 6, needs 3 bits
def entropy_byte_xor(val):
    v = val & 0xFF
    return 0 if v == 0 else math.ceil(math.log2(v + 1))


# total bit cost for a whole cache line (subtraction version)
def entropy_line(byteline):
    return sum(entropy_byte(b) for b in byteline)


# total bit cost for a whole cache line (XOR version)
def entropy_line_xor(byteline):
    return sum(entropy_byte_xor(b) for b in byteline)


# how many bytes are zero - more zeros = easier to compress
def zero_count(byteline):
    return sum(1 for b in byteline if (b & 0xFF) == 0)


# subtract two lines byte by byte, mod 256 (no carry between bytes)
def byte_subtract(a, b):
    return [(a[i] - b[i]) & 0xFF for i in range(len(a))]


# add two lines byte by byte, mod 256 - this is how decompress
def byte_add(a, b):
    return [(a[i] + b[i]) & 0xFF for i in range(len(a))]


# XOR two lines byte by byte
def byte_xor(a, b):
    return [a[i] ^ b[i] for i in range(len(a))]


# check that can perfectly recover the original from the compressed delta
def verify_reconstruction(original, delta, bases, mode='sub'):
    result = list(delta)
    for base in bases:
        if mode == 'sub':
            result = byte_add(result, base)
        else:
            result = byte_xor(result, base)
    return result == original


# ---- the 3 compression methods ----

# method 1: XOR cache - just XOR the target with whoever gives the smallest result
# always uses exactly 1 partner line
def xor_best(lines, target_idx):
    target = lines[target_idx]
    best_cost = entropy_line_xor(target)  # baseline: no partner
    best_delta = list(target)
    best_partner = -1

    for i in range(len(lines)):
        if i == target_idx:
            continue
        xored = byte_xor(target, lines[i])
        cost = entropy_line_xor(xored)
        if cost < best_cost:
            best_cost = cost
            best_delta = list(xored)
            best_partner = i

    return {
        'method': 'xor',
        'entropy': best_cost,
        'N': 1 if best_partner >= 0 else 0,
        'bases': [best_partner] if best_partner >= 0 else [],
        'delta': best_delta,
        'zeros': zero_count(best_delta)
    }


# method 2: greedy - find the single best base first, then keep adding more
# bases as long as each one makes the result smaller
def greedy_search(lines, target_idx):
    target = lines[target_idx]
    orig_cost = entropy_line(target)

    # first pass: find the single best base out of all candidates
    best_single_cost = orig_cost
    best_single_idx = -1
    for i in range(len(lines)):
        if i == target_idx:
            continue
        cost = entropy_line(byte_subtract(target, lines[i]))
        if cost < best_single_cost:
            best_single_cost = cost
            best_single_idx = i

    # if nothing helped, return the original unchanged
    if best_single_idx == -1:
        return {
            'method': 'greedy',
            'entropy': orig_cost,
            'N': 0,
            'bases': [],
            'delta': list(target),
            'zeros': zero_count(target)
        }

    current_delta = byte_subtract(target, lines[best_single_idx])
    current_cost = best_single_cost
    included = [best_single_idx]

    # second pass: keep subtracting more lines if they help
    for i in range(len(lines)):
        if i == target_idx or i == best_single_idx:
            continue
        candidate = byte_subtract(current_delta, lines[i])
        new_cost = entropy_line(candidate)
        if new_cost < current_cost:
            current_cost = new_cost
            current_delta = candidate
            included.append(i)

    return {
        'method': 'greedy',
        'entropy': current_cost,
        'N': len(included),
        'bases': included,
        'delta': current_delta,
        'zeros': zero_count(current_delta)
    }


# method 3: SAD (sum of absolute differences) - cheap hardware-friendly way
# to pick the best base. same idea as video codec motion estimation.
# no log2 needed, just add up absolute differences and pick the smallest
def sad_delta_search(lines, target_idx):
    target = lines[target_idx]
    best_sad = float('inf')
    best_partner = -1

    for i in range(len(lines)):
        if i == target_idx:
            continue
        # treat each byte difference as signed then sum the absolutes
        sad = sum(
            abs(((target[j] - lines[i][j]) + 128) % 256 - 128)
            for j in range(len(target))
        )
        if sad < best_sad:
            best_sad = sad
            best_partner = i

    if best_partner >= 0:
        delta = byte_subtract(target, lines[best_partner])
        ent = entropy_line(delta)
        return {
            'method': 'sad',
            'entropy': ent,
            'N': 1,
            'bases': [best_partner],
            'delta': delta,
            'zeros': zero_count(delta),
            'sad': best_sad
        }
    else:
        return {
            'method': 'sad',
            'entropy': entropy_line(target),
            'N': 0,
            'bases': [],
            'delta': list(target),
            'zeros': zero_count(target),
            'sad': 0
        }


# ---- image helpers ----

# pull out a block of pixel rows from the image as a list of byte lists
# each row = one 64-byte "cache line"
def extract_region_lines(img_gray, x0, y0, num_lines=8, line_width=64):
    w, h = img_gray.size
    lines = []
    for row in range(num_lines):
        y = y0 + row
        if y >= h:
            break
        line = []
        for col in range(line_width):
            x = x0 + col
            line.append(img_gray.getpixel((x, y)) if x < w else 0)
        lines.append(line)
    return lines


# pick 7 interesting spots in the image to test
def get_regions_for_image(w, h):
    if w >= 256 and h >= 256:
        specs = [
            ("Smooth skin (cheek)",       0.47, 0.55),
            ("Hat brim (sharp edge)",     0.35, 0.25),
            ("Feather texture (busy)",    0.20, 0.43),
            ("Background (flat)",         0.68, 0.04),
            ("Mirror edge (vertical)",    0.78, 0.59),
            ("Hair (medium texture)",     0.59, 0.39),
            ("Middle of image (general)", 0.00, 0.50),
        ]
    else:
        specs = [
            ("Top-left region",      0.05, 0.05),
            ("Top-right region",     0.60, 0.05),
            ("Center region",        0.25, 0.45),
            ("Mid-left region",      0.05, 0.40),
            ("Mid-right region",     0.60, 0.55),
            ("Bottom-left region",   0.10, 0.75),
            ("Bottom-center region", 0.30, 0.70),
        ]

    regions = []
    for name, xf, yf in specs:
        x0 = min(int(xf * w), w - 64) if w >= 64 else 0
        y0 = min(int(yf * h), h - 8)  if h >= 8  else 0
        regions.append((name, x0, y0))
    return regions


# ---- run all 3 methods on a set of lines and print results ----
def analyze_lines(lines, name):
    if len(lines) < 2:
        print(f"  Not enough lines for {name}")
        return None
    if sum(1 for line in lines if any(b != 0 for b in line)) < 2:
        print(f"  Skipping {name} -- all zeros (out of bounds)")
        return None

    print(f"\n{'-' * 68}")
    print(f"  REGION: {name}")
    print(f"  Lines: {len(lines)}, Width: {len(lines[0])} bytes")
    print(f"{'-' * 68}")

    # show first 12 bytes of each line so  can see what the data looks like
    for i, line in enumerate(lines):
        first = ' '.join(f'{b:02X}' for b in line[:12])
        print(f"  Line {i}: {first} ...  bitwidth_cost={entropy_line(line)} bits")

    results   = {'none': [], 'xor': [], 'greedy': [], 'sad': []}
    times_cmp = {'xor': [], 'greedy': [], 'sad': []}
    times_dcp = {'xor': [], 'greedy': [], 'sad': []}
    n_dist    = {'greedy': []}

    for t in range(len(lines)):
        results['none'].append(entropy_line(lines[t]))

        # time XOR compress
        t0 = time.perf_counter_ns()
        r_xor = xor_best(lines, t)
        times_cmp['xor'].append(time.perf_counter_ns() - t0)
        results['xor'].append(r_xor['entropy'])

        # time XOR decompress
        t0 = time.perf_counter_ns()
        if r_xor['bases']:
            byte_xor(r_xor['delta'], lines[r_xor['bases'][0]])
        times_dcp['xor'].append(time.perf_counter_ns() - t0)

        # time greedy compress
        t0 = time.perf_counter_ns()
        r_g = greedy_search(lines, t)
        times_cmp['greedy'].append(time.perf_counter_ns() - t0)
        results['greedy'].append(r_g['entropy'])
        n_dist['greedy'].append(r_g['N'])

        # time greedy decompress
        t0 = time.perf_counter_ns()
        tmp = list(r_g['delta'])
        for bi in r_g['bases']:
            tmp = byte_add(tmp, lines[bi])
        times_dcp['greedy'].append(time.perf_counter_ns() - t0)

        # time SAD compress
        t0 = time.perf_counter_ns()
        r_sad = sad_delta_search(lines, t)
        times_cmp['sad'].append(time.perf_counter_ns() - t0)
        results['sad'].append(r_sad['entropy'])

        # time SAD decompress
        t0 = time.perf_counter_ns()
        if r_sad['bases']:
            byte_add(r_sad['delta'], lines[r_sad['bases'][0]])
        times_dcp['sad'].append(time.perf_counter_ns() - t0)

        # make sure can actually get the original back
        ok_xor = verify_reconstruction(lines[t], r_xor['delta'],
                     [lines[i] for i in r_xor['bases']], mode='xor')
        ok_g   = verify_reconstruction(lines[t], r_g['delta'],
                     [lines[i] for i in r_g['bases']], mode='sub')
        ok_sad = verify_reconstruction(lines[t], r_sad['delta'],
                     [lines[i] for i in r_sad['bases']], mode='sub')

        if not ok_xor:
            print(f"  XOR reconstruction FAILED for line {t}!")
        if not ok_g:
            print(f"  Greedy reconstruction FAILED for line {t}!")
        if not ok_sad:
            print(f"  SAD reconstruction FAILED for line {t}!")

    # average times across all lines in this region
    n_lines = len(lines)
    avg_cmp_us = {m: sum(times_cmp[m]) / n_lines / 1_000 for m in times_cmp}
    avg_dcp_us = {m: sum(times_dcp[m]) / n_lines / 1_000 for m in times_dcp}

    # analytical op counts (just counting byte-level operations)
    K = n_lines - 1
    W = len(lines[0])
    avg_N_greedy = sum(n_dist['greedy']) / n_lines

    ops_cmp = {
        'xor':    K * (W + W) + W,           # K candidates, XOR + cost each
        'greedy': K * (W + W) + (K-1) * (W + W),  # phase1 + phase2
        'sad':    K * W + W,                  # K SAD passes + final subtract
    }
    ops_dcp = {
        'xor':    W,                               # 1 XOR
        'greedy': max(1, round(avg_N_greedy)) * W, # N adds
        'sad':    W,                               # 1 add
    }

    # print the results table
    orig_avg = sum(results['none']) / n_lines
    method_avgs = {}

    print(f"\n  {'Method':<30} {'AvgBWCost':>9} {'Reduction':>10} {'Ratio':>7}"
          f"  {'Cmp(micro-s)':>8}  {'CmpOps':>7}  {'Dcp(micro-s)':>8}  {'DcpOps':>7}")
    print(f"  {'-' * 98}")

    for method, label in [
        ('none',   'No compression'),
        ('xor',    'XOR Cache (N=1, unsigned BW)'),
        ('greedy', 'Byte delta (greedy)'),
        ('sad',    'SAD Delta (Sum of Abs Diff)'),
    ]:
        avg = sum(results[method]) / n_lines
        method_avgs[method] = avg
        red   = (1 - avg / orig_avg) * 100 if orig_avg > 0 else 0
        ratio = orig_avg / avg if avg > 0 else float('inf')

        if method == 'none':
            print(f"  {label:<30} {avg:>7.1f} b {red:>9.1f}% {ratio:>6.2f}x"
                  f"  {'--':>8}  {'--':>7}  {'--':>8}  {'--':>7}")
        else:
            print(f"  {label:<30} {avg:>7.1f} b {red:>9.1f}% {ratio:>6.2f}x"
                  f"  {avg_cmp_us[method]:>8.3f}  {ops_cmp[method]:>7d}"
                  f"  {avg_dcp_us[method]:>8.4f}  {ops_dcp[method]:>7d}")

    print(f"\n  N chosen (greedy): {n_dist['greedy']}  (avg {avg_N_greedy:.2f})")

    # show the best single example using SAD so can see actual delta bytes
    best_t = min(range(len(lines)),
                 key=lambda t: sad_delta_search(lines, t)['entropy'])
    r = sad_delta_search(lines, best_t)
    orig_ent = entropy_line(lines[best_t])

    print(f"\n  Best case: target=Line {best_t}, base=Line {r['bases'][0] if r['bases'] else '?'}")
    print(f"    Original:  {' '.join(f'{b:02X}' for b in lines[best_t][:16])} ...")
    print(f"    Delta:     {' '.join(f'{b:02X}' for b in r['delta'][:16])} ...")

    signed_str = ' '.join(
        f"{(b if b < 128 else b - 256):+d}"
        for b in [x & 0xFF for x in r['delta'][:16]]
    )
    print(f"    Signed:    {signed_str} ...")

    if r['entropy'] > 0:
        print(f"    BW Cost:   {r['entropy']} bits (from {orig_ent}), "
              f"ratio: {orig_ent/r['entropy']:.2f}x")
    else:
        print(f"    BW Cost:   0 bits (perfect compression)")

    return {
        'name': name, 'orig': orig_avg,
        'xor': method_avgs['xor'],
        'greedy': method_avgs['greedy'],
        'sad': method_avgs['sad'],
        'cmp_us':  avg_cmp_us,
        'dcp_us':  avg_dcp_us,
        'ops_cmp': ops_cmp,
        'ops_dcp': ops_dcp,
        'avg_N':   avg_N_greedy,
    }


# ---- main ----
def main():
    print("=" * 68)
    print("  DELTA CACHE COMPRESSION ON IMAGE DATA")
    print("=" * 68)

    if len(sys.argv) != 2:
        print("  Usage: python3 delta_cache_image_test.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"  ERROR: '{path}' not found")
        sys.exit(1)

    # load image and convert to grayscale (1 byte per pixel)
    img = Image.open(path).convert('L')
    w, h = img.size
    print(f"  Loaded: {path} ({w}x{h} -> grayscale)")

    if w < 64 or h < 16:
        print(f"\n  ERROR: Image too small ({w}x{h}). Need at least 64x16.")
        sys.exit(1)
    if w < 128 or h < 128:
        print(f"  WARNING: Small image ({w}x{h}). Use 256+ for best results.\n")

    # run analysis on each region
    regions = get_regions_for_image(w, h)
    all_results = []
    for name, x0, y0 in regions:
        print(f"\n  Extracting: {name} at ({x0}, {y0})")
        lines = extract_region_lines(img, x0, y0, num_lines=8, line_width=64)
        r = analyze_lines(lines, name)
        if r:
            all_results.append(r)

    if not all_results:
        print("\n  No valid regions found.")
        return

    # final summary across all regions
    print("\n" + "=" * 68)
    print("  SUMMARY: ALL REGIONS")
    print("=" * 68)
    print(f"\n  {'Region':<30} {'Orig':>5} {'XOR':>6} {'Grdy':>6} {'SAD':>6}")
    print(f"  {'-' * 55}")

    for r in all_results:
        def ratio(m):
            return r['orig'] / r[m] if r[m] > 0 else float('inf')
        print(f"  {r['name']:<30} {r['orig']:>5.0f} "
              f"{ratio('xor'):>5.2f}x "
              f"{ratio('greedy'):>5.2f}x "
              f"{ratio('sad'):>5.2f}x")

    n = len(all_results)
    avg_orig = sum(r['orig'] for r in all_results) / n
    print(f"  {'-' * 55}")
    avg_line = f"  {'AVERAGE':<30} {avg_orig:>5.0f} "
    for method in ['xor', 'greedy', 'sad']:
        avg_ent = sum(r[method] for r in all_results) / n
        avg_ratio = avg_orig / avg_ent if avg_ent > 0 else float('inf')
        avg_line += f"{avg_ratio:>5.2f}x "
    print(avg_line)

    # timing summary across all regions
    print(f"\n  {'Method':<30} {'Cmp micro-s':>8}  {'CmpOps':>7}  {'Dcp micro-s':>8}  {'DcpOps':>7}  Note")
    print(f"  {'-' * 80}")

    method_meta = [
        ('xor',    'XOR Cache (N=1)',        'symmetric, N=1 always'),
        ('greedy', 'Byte delta (greedy)',     f"adaptive N, avg N={sum(r['avg_N'] for r in all_results)/n:.2f}"),
        ('sad',    'SAD Delta (Sum Abs Diff)', 'L1 proxy, N=1 always'),
    ]
    for m, label, note in method_meta:
        avg_cmp     = sum(r['cmp_us'][m]  for r in all_results) / n
        avg_dcp     = sum(r['dcp_us'][m]  for r in all_results) / n
        avg_cmp_ops = sum(r['ops_cmp'][m] for r in all_results) / n
        avg_dcp_ops = sum(r['ops_dcp'][m] for r in all_results) / n
        print(f"  {label:<30} {avg_cmp:>8.3f}  {avg_cmp_ops:>7.0f}  {avg_dcp:>8.4f}  {avg_dcp_ops:>7.0f}  {note}")

    print(f"\n  (micro-s = measured Python wall-clock per line; Ops = byte-level arithmetic ops, analytical)")


if __name__ == '__main__':
    main()

