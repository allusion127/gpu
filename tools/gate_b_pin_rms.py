# -*- coding: utf-8 -*-
# BOC pin power RMS/max %% comparison: RASBERY h5 vs MASTER PPI (BOC).
# Standalone extraction of the BOC-only logic from make_v2_master_cmp.py
# (no plotting / matplotlib dependency), for the v2 re-freeze protocol.
import re
import sys
import numpy as np
import h5py

if len(sys.argv) != 3:
    print("usage: pin_boc_cmp.py <rasbery.h5> <kngr_mas_ppi_boc.txt>")
    sys.exit(2)

H5 = sys.argv[1]
PPI = sys.argv[2]

f = h5py.File(H5, "r")

kbc, kec = int(f["geometry"]["kbc"][()]), int(f["geometry"]["kec"][()])
ij = f["geometry"]["ijtola"][()]
nxa_g = int(f["geometry"]["nxa"][()])
nya_g = int(f["geometry"]["nya"][()])
ij = ij.reshape(nya_g, nxa_g)
ras_map = {}
for jj in range(ij.shape[0]):
    for ii in range(ij.shape[1]):
        la = int(ij[jj, ii])
        if la >= 0:
            ras_map[(jj, ii)] = la

XL = "ABCDEFGHJKLMNPRST"


def parse_ppi(path):
    ppi = open(path, errors="replace").read()
    heads = list(re.finditer(r"^\s*FANAME\s+([A-Z]+)\s+(\d+)\s+(\d+)\s", ppi, re.M))
    mas_pin = {}
    for hi, h in enumerate(heads):
        body = ppi[h.end(): heads[hi + 1].start() if hi + 1 < len(heads) else len(ppi)]
        col = XL.index(h.group(1)) - XL.index("J")
        row = int(h.group(2)) - 9
        m = re.search(r"PIN 3-D POWER DISTRIBUTION[^\n]*\n(.*)\Z", body, re.S)
        if not m:
            continue
        seg = m.group(1)
        stop = re.search(r"\n[A-Z][A-Z \-0-9]+ DISTRIBUTION|\n\*{10,}", seg)
        if stop:
            seg = seg[: stop.start()]
        nums = np.array([float(x) for x in re.findall(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?", seg)])
        npl = nums.size // 256
        if npl == 0:
            continue
        mas_pin[(row, col)] = nums[: npl * 256].reshape(npl, 16, 16).mean(axis=0)
    return mas_pin


def pin_err_stats(mas_pin, ras_pin):
    mas_all, ras_all = [], []
    for (r, c), mp in mas_pin.items():
        la = ras_map.get((r, c))
        if la is None:
            continue
        rp = ras_pin[la]
        if not np.any(rp > 0):
            continue
        mas_all.append(mp)
        ras_all.append(rp)
    mnorm = np.mean([a[a > 0].mean() for a in mas_all])
    rnorm = np.mean([a[a > 0].mean() for a in ras_all])
    errs = []
    for (r, c), mp in mas_pin.items():
        la = ras_map.get((r, c))
        if la is None or r > 8 or c > 8:
            continue
        rp = ras_pin[la]
        if not np.any(rp > 0):
            continue
        e = np.where((mp > 0.05) & (rp > 0), (rp / rnorm) / (mp / mnorm) - 1.0, np.nan) * 100
        errs.append(e[np.isfinite(e)])
    allerr = np.concatenate(errs)
    return float(np.sqrt(np.nanmean(allerr ** 2))), float(np.nanmax(np.abs(allerr)))


def ras_pin_at(step):
    pp = f[f"steps/{step}/pin_power"][()]
    return pp[kbc:kec].mean(axis=0)


mas_boc = parse_ppi(PPI)
rms_boc, max_boc = pin_err_stats(mas_boc, ras_pin_at("0001"))
print("BOC pin: rms %.3f%% max %.2f%%" % (rms_boc, max_boc))
