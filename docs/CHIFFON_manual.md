# CHIFFON Input Manual

> Note: some RHST examples below use legacy keyed rod-history terminology. The
> current keyless SPCT+RDPL+RHST validation setup is documented in
> `test/9-1_IISC/docs/keyless_iisc_rhst_methodology_ko.md`.

CHIFFON builds cross-section delta functions from HGC reference, branch, extra-reference, and rod-history data.

Only two function types are supported:

- `spline`: piecewise polynomial interpolation. With `order: 1`, this is linear interpolation.
- `polynomial` or `poly`: global polynomial interpolation.

The supported correction families are:

- `bppm`
- `tful`
- `dmod`
- `state_delta` for `avg_tmod` and `rated_power`
- `isotope_vector_absolute`

## Branch Deltas

```json
{
  "settings": {
    "bppm": {"apply": true, "order": 1, "type": "spline"},
    "tful": {"apply": true, "order": 1, "type": "spline"},
    "dmod": {
      "apply": true,
      "order": 1,
      "type": "spline",
      "pre_remove": ["bppm"]
    }
  }
}
```

`pre_remove` subtracts already-fitted corrections before fitting the current correction. This is useful when a branch or extra-reference file contains more than one physical effect.

## Averaged State Delta

`state_delta` uses extra reference HGC files. Only reference points are used; branch points in extra files are ignored.

```json
{
  "settings": {
    "state_delta": [
      {
        "apply": true,
        "field": "avg_tmod",
        "order": 1,
        "type": "spline",
        "pre_remove": ["dmod"]
      },
      {
        "apply": true,
        "field": "rated_power",
        "order": 1,
        "type": "spline",
        "pre_remove": ["dmod"]
      }
    ]
  }
}
```

Supported fields:

- `avg_tmod`: `volume_average(Tmod_current) - Tmod_ref`
- `rated_power`: `PDEN_extra / PDEN_ref - 1` when fitting, and `current_power_rate - 1` at runtime

## Isotope Vector Absolute

`isotope_vector_absolute` fits absolute isotope-density coordinates. Each
isotope in `isotopes` gets a linear term:

```text
x_i = N_i
```

Isotopes used only in
`cross-terms` are still registered for that term, but they do not get their own
linear term unless they are also listed in `isotopes`. A two-isotope entry in
`cross-terms` adds a ratio coordinate:

```text
x_{A/B} = N_A / N_B
```

If the denominator is below `1.0e-10`, the ratio coordinate is set to zero for
that fit or runtime correction point.

Entries with three or more isotopes remain product terms.

```json
{
  "settings": {
    "isotope_vector_absolute": {
      "apply": true,
      "isotopes": ["Pu239", "Xe135", "Pm149", "Gd"],
      "cross-terms": [["Xe135", "Pm149"]],
      "pre_remove": ["dmod"]
    }
  }
}
```

This fits `Pu239`, `Xe135`, `Pm149`, `Gd`, and `Xe135/Pm149`. Isotope names may be HGC ids such as `942390` or common symbols such as `Pu239`; lumped gadolinium may be written as `Gd`.

CHIFFON still fits cross-section residuals against the matching main reference:

```text
y = XS_point - XS_reference
```

The isotope density coordinate itself is not differenced against the reference.
This is useful for very small or short-lived inventories where dividing by, or
subtracting, the reference density would amplify depletion-code differences.

`extra` files fit isotope coefficients around the main reference state.
`rod_history` files are split into two pieces:

- a trajectory offset between the main reference and the rod-history reference;
- local isotope coefficients around that rod-history reference.

At runtime RASBERY applies:

```text
Delta XS = trajectory_offset + isotope_coeff(trajectory ctype, current ctype) * x
```

For `rod_history`, the object key such as `CR1` means the depletion trajectory
type. A rod-history HGC is assumed to have been depleted rodded. Its reference
points, which usually have HGC ctype 0, are mapped to the trajectory ctype.
Its `CR*` branch points are mapped to current ctype 0 because they represent the
rod-out current state from the rodded depletion basis.

Object entries may set `current_ctype` or `solve_ctype` to override this
automatic mapping. Use `"hgc"` to preserve the ctype written in the HGC title.

```json
{
  "fuels": {
    "A1": {
      "main": "dec_FA_A1_0101.HGC",
      "rod_history": {
        "CR1": [
          "dec_FA_A1_CR1_history_0101.HGC",
          "dec_FA_A1_CR1_low_power_history_0101.HGC"
        ],
        "CR2": "dec_FA_A1_CR2_history_0101.HGC"
      }
    }
  }
}
```

State history coefficients are stored by the current control-rod type.

## Rod Depletion

`rod depletion` supplies rod-material delta XS as a function of fluence. The HGC
burnup scalar is interpreted as fluence for this input, and CHIFFON fits one
`fluence -> delta XS` table per HGC `ctype`. Fuel burnup is not used for this
correction.

```json
{
  "settings": {
    "rod depletion": {"apply": true, "order": 1, "type": "spline"}
  },
  "rod depletion": "dec_rod_depletion_0101.HGC"
}
```

## Burnup Interpolation

All branch and extra-reference delta corrections are stored by burnup. RASBERY linearly interpolates between lower and upper burnup correction entries at runtime:

```text
delta = (1 - f) * delta_lo + f * delta_hi
```

This applies to `bppm`, `tful`, `dmod`, `state_delta`, and `isotope_vector_absolute`.

## Example

```json
{
  "settings": {
    "discontinuity factor": true,
    "bppm": {"apply": true, "order": 1, "type": "spline"},
    "tful": {"apply": true, "order": 1, "type": "spline"},
    "dmod": {
      "apply": true,
      "order": 1,
      "type": "spline",
      "pre_remove": ["bppm"]
    },
    "isotope_vector_absolute": {
      "apply": false,
      "isotopes": ["Pu239", "Xe135", "Pm149"],
      "cross-terms": [["Xe135", "Pm149"]],
      "pre_remove": ["dmod"]
    },
    "state_delta": [
      {
        "apply": true,
        "field": "avg_tmod",
        "order": 1,
        "type": "spline",
        "pre_remove": ["dmod"]
      },
      {
        "apply": true,
        "field": "rated_power",
        "order": 1,
        "type": "spline",
        "pre_remove": ["dmod"]
      }
    ]
  },
  "fuels": {
    "W1": {
      "main": "WH17_High_0101.HGC",
      "extra": [
        "WH17_540_0101.HGC",
        "WH17_600_0101.HGC",
        "WH17_P50_0101.HGC"
      ],
      "rod_history": [
        {"file": "WH17_CR1_history_0101.HGC", "control_rod_type": 1}
      ]
    }
  }
}
```
