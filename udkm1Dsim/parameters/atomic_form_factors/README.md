# Atomic Form Factors

## `chantler/` and `henke/`

Energy-dependent atomic form factors $f_1(E)$ and $f_2(E)$, see the README
files in the according folders.

## `cromermann.txt`

Coefficients of the analytical approximation of the atomic scattering factors
by Cromer and Mann for neutral atoms and ions

$$
f_{CM}(\sin(\theta)/\lambda) = \sum_{i=1}^4 a_i \exp\left(-b_i \left(\sin(\theta)/\lambda\right)^2\right) + c
$$

with $\sin(\theta)/\lambda = q/4\pi$ in Å⁻¹, which is valid for
$\sin(\theta)/\lambda < 2$ Å⁻¹, see `Atom.get_cm_atomic_form_factor()`.

### Source

International Tables for Crystallography, Volume C, Table 6.1.1.4,
_Coefficients for analytical approximation to the scattering factors of
Tables 6.1.1.1 and 6.1.1.3_, see
[P. J. Brown, A. G. Fox, E. N. Maslen, M. A. O'Keefe, and B. T. M. Willis, ch. 6.1](https://it.iucr.org/Cb/ch6o1v0001/),
which is a reprint of Table 2.2B of the International Tables for X-ray
Crystallography, Volume IV, The Kynoch Press, Birmingham (1974). The
coefficients are based on D. T. Cromer and J. B. Mann,
[Acta Cryst. A **24**, 321 (1968)](https://doi.org/10.1107/S0567739468000550).

### Format

Tab-separated columns with one header line:

| column | description |
|---|---|
| `E` | symbol of the element |
| `Z` | atomic number |
| `I` | ionicity |
| `a1` .. `a4` | coefficients $a_i$ |
| `b1` .. `b4` | coefficients $b_i$ [Å²] |
| `c` | coefficient $c$ |

`Atom.read_cromer_mann_coeff()` returns the first entry for the atomic number
and ionicity of an atom. Some elements have additional entries:

| entry | description | label in Table 6.1.1.4 |
|---|---|---|
| first `H` with `I = 0` | hydrogen with the form factor of Stewart, Davidson & Simpson (SDS) | `H` |
| `H.` | isolated hydrogen atom | `H` (isolated atom) |
| second `C` with `I = 0` | valence state of carbon | `Cval` |
| `Si` with `I = 99` | valence state of silicon | `Sival` |
