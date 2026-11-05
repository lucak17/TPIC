#!/usr/bin/env python3
"""Analyze a seeded parallel whistler wave and compare it with cold theory.

Usage from the repository root:

	python3 tools/postprocessing/analyze_whistler.py

By default the script analyzes the current directory and writes:

	whistler_analysis.png

To analyze another run directory or choose the phase/growth fit window:

	python3 tools/postprocessing/analyze_whistler.py \
		--run-dir /path/to/run \
		--fit-start 5.0 \
		--fit-end 30.0

The fit-window values are in ion time units, ``omega_pi * t``.  The script
assumes the parallel whistler of the input deck, ``B0 = B0x xhat`` and
``k = kx xhat``, and analyzes ``b_R = By - i Bz``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # source-checkout imports
from tpic_tools.inputs import periodic_axes, read_input
from tpic_tools.fields import field_data_exists, open_field_reader, transverse_mean


TWO_PI = 2.0 * math.pi




def first_float(values: dict[str, list[str]], key: str, default: float | None = None) -> float:
	if key not in values:
		if default is None:
			raise KeyError(f"missing required input key {key}")
		return default
	return float(values[key][0])


def first_int(values: dict[str, list[str]], key: str, default: int | None = None) -> int:
	return int(first_float(values, key, default))


def first_bool(values: dict[str, list[str]], key: str, default: bool = False) -> bool:
	if key not in values:
		return default
	token = values[key][0].strip().lower()
	return token in {"1", "true", "yes", "on"}


def first_string(values: dict[str, list[str]], key: str, default: str) -> str:
	if key not in values:
		return default
	return values[key][0].strip()


def whistler_seed_mode(values: dict[str, list[str]]) -> str:
	token = first_string(values, "WhistlerSeedMode", "eigenmode").lower().replace("-", "_")
	if token in {"eigenmode", "eigen_mode"}:
		return "eigenmode"
	if token in {"magnetic", "magnetic_only", "b_only"}:
		return "magnetic_only"
	if token in {"noise", "magnetic_noise"}:
		return "noise"
	raise ValueError(f"unknown WhistlerSeedMode: {first_string(values, 'WhistlerSeedMode', 'eigenmode')}")


def qom_mass_ratio(values: dict[str, list[str]]) -> float:
	qoms = [float(token) for token in values["qom"]]
	electron_qom = next((abs(qom) for qom in qoms if qom < 0.0), None)
	ion_qom = next((qom for qom in qoms if qom > 0.0), None)
	if electron_qom is None or ion_qom is None:
		raise ValueError("Whistler analysis requires one electron and one ion qom")
	return electron_qom / ion_qom


def cold_whistler_approx_frequency(mu: float, b0: float, k_di: float, light_speed: float) -> float:
	ck = light_speed * k_di
	return b0 * ck * ck / (1.0 + ck * ck / mu)


def cold_whistler_residual(
	omega: float,
	mu: float,
	b0: float,
	k_di: float,
	light_speed: float,
) -> float:
	lhs = light_speed * light_speed * k_di * k_di / (omega * omega)
	rhs = (
		1.0
		- mu / (omega * (omega - mu * b0))
		- 1.0 / (omega * (omega + b0))
	)
	return lhs - rhs


def cold_whistler_exact_frequency(
	mu: float,
	b0: float,
	k_di: float,
	light_speed: float,
) -> float:
	lower = b0 * (1.0 + 1.0e-8)
	upper = mu * b0 * (1.0 - 1.0e-8)
	left_omega = lower
	left_value = cold_whistler_residual(left_omega, mu, b0, k_di, light_speed)

	for sample in range(1, 4097):
		fraction = sample / 4096.0
		right_omega = lower + fraction * (upper - lower)
		right_value = cold_whistler_residual(right_omega, mu, b0, k_di, light_speed)
		if not math.isfinite(left_value) or not math.isfinite(right_value):
			left_omega = right_omega
			left_value = right_value
			continue
		if left_value == 0.0:
			return left_omega
		if left_value * right_value <= 0.0:
			bracket_left = left_omega
			bracket_right = right_omega
			value_left = left_value
			for _ in range(100):
				mid_omega = 0.5 * (bracket_left + bracket_right)
				mid_value = cold_whistler_residual(mid_omega, mu, b0, k_di, light_speed)
				if abs(mid_value) < 1.0e-13:
					return mid_omega
				if value_left * mid_value <= 0.0:
					bracket_right = mid_omega
				else:
					bracket_left = mid_omega
					value_left = mid_value
			return 0.5 * (bracket_left + bracket_right)

		left_omega = right_omega
		left_value = right_value

	raise RuntimeError("unable to find the cold right-hand whistler root")


def whistler_dispersion_curve(
	mode_max: int,
	lx: float,
	mu: float,
	b0: float,
	light_speed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	modes = np.arange(1, mode_max + 1)
	k_di = TWO_PI * modes / lx
	k_de = k_di / math.sqrt(mu)
	omega_approx = np.asarray([
		cold_whistler_approx_frequency(mu, b0, k_value, light_speed)
		for k_value in k_di
	])
	omega_exact = []
	for k_value in k_di:
		try:
			omega_exact.append(cold_whistler_exact_frequency(mu, b0, k_value, light_speed))
		except RuntimeError:
			omega_exact.append(float("nan"))
	return k_di, k_de, omega_approx, np.asarray(omega_exact)


def read_scalar(reader, name: str, step: int) -> float:
	return float(np.asarray(reader.read(name, step_selection=[step, 1])).ravel()[0])


def x_profile(reader, name: str, step: int, periodic: tuple[bool, bool, bool]) -> np.ndarray:
	# The mean over y and z along the periodic x, without its last node, the image of the first.
	return transverse_mean(reader.read(name, step_selection=[step, 1]), periodic)[:-1]


def fft_mode(profile: np.ndarray, mode_x: int) -> complex:
	if mode_x < 0 or mode_x >= profile.size:
		raise ValueError(f"mode {mode_x} is outside profile length {profile.size}")
	return np.fft.fft(profile)[mode_x] / profile.size


def read_component_modes(
	path: Path,
	variables: tuple[str, ...],
	mode_x: int,
	periodic: tuple[bool, bool, bool],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
	histories = {name: [] for name in variables}
	times = []

	with open_field_reader(path) as reader:
		available = reader.available_variables()
		missing = [name for name in variables if name not in available]
		if missing:
			raise KeyError(f"{path} is missing variables: {', '.join(missing)}")

		for step in range(reader.num_steps()):
			times.append(read_scalar(reader, "time", step))
			for name in variables:
				histories[name].append(fft_mode(x_profile(reader, name, step, periodic), mode_x))

	return np.asarray(times), {name: np.asarray(values) for name, values in histories.items()}


def read_b_right_history(
	path: Path,
	mode_x: int,
	max_spectrum_mode: int,
	periodic: tuple[bool, bool, bool],
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
	times = []
	by_mode = []
	bz_mode = []
	right_mode = []
	left_mode = []
	first_spectrum = None
	last_spectrum = None

	with open_field_reader(path) as reader:
		for step in range(reader.num_steps()):
			by_profile = x_profile(reader, "By", step, periodic)
			bz_profile = x_profile(reader, "Bz", step, periodic)
			right_profile = by_profile - 1j * bz_profile
			left_profile = by_profile + 1j * bz_profile
			spectrum = np.fft.fft(right_profile) / right_profile.size
			positive_limit = min(max_spectrum_mode + 1, spectrum.size // 2 + 1)

			if first_spectrum is None:
				first_spectrum = spectrum[:positive_limit].copy()
			last_spectrum = spectrum[:positive_limit].copy()

			times.append(read_scalar(reader, "time", step))
			by_mode.append(fft_mode(by_profile, mode_x))
			bz_mode.append(fft_mode(bz_profile, mode_x))
			right_mode.append(spectrum[mode_x])
			left_mode.append(fft_mode(left_profile, mode_x))

	if first_spectrum is None or last_spectrum is None:
		raise ValueError(f"{path} contains no steps")

	modes = {
		"By": np.asarray(by_mode),
		"Bz": np.asarray(bz_mode),
		"b_right": np.asarray(right_mode),
		"b_left": np.asarray(left_mode),
	}
	return np.asarray(times), modes, first_spectrum, last_spectrum


def fit_phase_and_growth(
	time_pi: np.ndarray,
	mode: np.ndarray,
	fit_start: float,
	fit_end: float,
) -> tuple[float, float, float, float, np.ndarray]:
	keep = (time_pi >= fit_start) & (time_pi <= fit_end) & (np.abs(mode) > 0.0)
	if np.count_nonzero(keep) < 3:
		raise ValueError("fit window must contain at least three nonzero samples")

	fit_time = time_pi[keep]
	phase = np.unwrap(np.angle(mode))
	phase_slope, phase_intercept = np.polyfit(fit_time, phase[keep], 1)
	log_amp_slope, log_amp_intercept = np.polyfit(fit_time, np.log(np.abs(mode[keep])), 1)
	return -phase_slope, phase_intercept, log_amp_slope, log_amp_intercept, phase


def safe_median_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
	keep = np.abs(denominator) > 0.0
	if not np.any(keep):
		return float("nan")
	return float(np.median(np.abs(numerator[keep]) / np.abs(denominator[keep])))


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
	return (angle + math.pi) % (2.0 * math.pi) - math.pi


def mode_power_fraction(spectrum: np.ndarray, mode_x: int) -> float:
	if mode_x >= spectrum.size:
		return float("nan")
	power = np.abs(spectrum) ** 2
	total = power[1:].sum()
	if total == 0.0:
		return float("nan")
	return float(power[mode_x] / total)


def plot_analysis(
	output_path: Path,
	time_pi: np.ndarray,
	b_right: np.ndarray,
	phase: np.ndarray,
	first_spectrum: np.ndarray,
	last_spectrum: np.ndarray,
	mode_x: int,
	delta_b: float,
	k_de: float,
	omega_theory: float,
	omega_pic: float,
	gamma_pic: float,
	dispersion_k_de: np.ndarray,
	dispersion_omega_approx: np.ndarray,
	dispersion_omega_exact: np.ndarray,
	phase_intercept: float,
	log_amp_intercept: float,
	fit_start: float,
	fit_end: float,
) -> None:
	fit_time = np.linspace(time_pi.min(), time_pi.max(), 256)
	phase_fit = phase_intercept - omega_pic * fit_time
	phase_anchor = np.interp(fit_start, time_pi, phase)
	phase_theory = phase_anchor - omega_theory * (fit_time - fit_start)

	amp_scale = delta_b if delta_b > 0.0 else 1.0
	energy_scale = amp_scale * amp_scale
	amp_fit = np.exp(log_amp_intercept + gamma_pic * fit_time) / amp_scale
	energy_fit = amp_fit * amp_fit
	phase_residual = phase - (phase_intercept - omega_theory * time_pi)

	fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0))

	ax = axes[0, 0]
	ax.plot(time_pi, np.abs(b_right) / amp_scale, "o-", label=r"PIC $|\hat b_R|/\delta B$")
	ax.plot(fit_time, amp_fit, "--", label=rf"fit $\gamma={gamma_pic:.3e}$")
	ax.axhline(1.0, color="0.35", linestyle=":", label="seed amplitude")
	ax.axvspan(fit_start, fit_end, color="0.92", zorder=0)
	ax.set_xlabel(r"$\omega_{pi} t$")
	ax.set_ylabel("normalized amplitude")
	ax.grid(True, alpha=0.3)
	ax.legend()

	ax = axes[0, 1]
	ax.plot(time_pi, phase, "o-", label="PIC unwrapped phase")
	ax.plot(fit_time, phase_fit, "--", label=rf"fit $\omega={omega_pic:.6f}$")
	ax.plot(fit_time, phase_theory, ":", linewidth=2.0, label=rf"cold theory $\omega={omega_theory:.6f}$")
	ax.axvspan(fit_start, fit_end, color="0.92", zorder=0)
	ax.set_xlabel(r"$\omega_{pi} t$")
	ax.set_ylabel(r"$\arg \hat b_R$")
	ax.grid(True, alpha=0.3)
	ax.legend()

	ax = axes[0, 2]
	ax.plot(dispersion_k_de, dispersion_omega_approx, "--", label="approx cold whistler")
	ax.plot(dispersion_k_de, dispersion_omega_exact, "-", label="exact cold R branch")
	ax.plot(k_de, omega_theory, "o", color="tab:green", label="seed theory")
	ax.plot(k_de, omega_pic, "x", markersize=9, markeredgewidth=2, color="tab:red", label="PIC phase fit")
	ax.set_xlabel(r"$k d_e$")
	ax.set_ylabel(r"$\omega/\omega_{pi}$")
	ax.set_title(rf"dispersion, mode $m={mode_x}$")
	ax.grid(True, alpha=0.3)
	ax.legend()

	ax = axes[1, 0]
	ax.semilogy(time_pi, np.abs(b_right) ** 2 / energy_scale, "o-", label=r"PIC $|\hat b_R|^2/\delta B^2$")
	ax.semilogy(fit_time, energy_fit, "--", label="growth fit")
	ax.axhline(1.0, color="0.35", linestyle=":", label="seed energy")
	ax.axvspan(fit_start, fit_end, color="0.92", zorder=0)
	ax.set_xlabel(r"$\omega_{pi} t$")
	ax.set_ylabel("normalized seeded-mode energy")
	ax.grid(True, which="both", alpha=0.3)
	ax.legend()

	ax = axes[1, 1]
	modes = np.arange(first_spectrum.size)
	ax.semilogy(modes, np.abs(first_spectrum) / amp_scale, "o-", label="first output")
	ax.semilogy(modes, np.abs(last_spectrum) / amp_scale, "s-", label="last output")
	ax.axvline(mode_x, color="0.35", linestyle=":", label=f"seed mode m={mode_x}")
	ax.set_xlabel("x Fourier mode")
	ax.set_ylabel(r"$|\hat b_R(m)|/\delta B$")
	ax.grid(True, which="both", alpha=0.3)
	ax.legend()

	ax = axes[1, 2]
	ax.plot(time_pi, wrap_to_pi(phase_residual), "o-")
	ax.axhline(0.0, color="0.35", linestyle=":")
	ax.axvspan(fit_start, fit_end, color="0.92", zorder=0)
	ax.set_xlabel(r"$\omega_{pi} t$")
	ax.set_ylabel("phase residual vs theory [rad]")
	ax.grid(True, alpha=0.3)

	fig.suptitle("Parallel whistler validation")
	fig.tight_layout()
	fig.savefig(output_path, dpi=200)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--run-dir", type=Path, default=Path("."))
	parser.add_argument("--input", type=Path, default=None)
	parser.add_argument("--fit-start", type=float, default=None, help=r"fit start in omega_pi t")
	parser.add_argument("--fit-end", type=float, default=None, help=r"fit end in omega_pi t")
	parser.add_argument("--output", type=Path, default=None)
	parser.add_argument("--max-spectrum-mode", type=int, default=32)
	parser.add_argument("--max-dispersion-mode", type=int, default=None)
	args = parser.parse_args()

	input_path = args.input or (args.run_dir / "whistler2D.inp")
	params = read_input(input_path)
	save_dir = params["SaveDirName"][0]
	data_dir = args.run_dir / save_dir

	periodic = periodic_axes(params)
	if not periodic[0]:
		raise ValueError(f"{input_path}: the mode analysis needs a box periodic in x")
	mode_x = first_int(params, "WhistlerModeX")
	lx = first_float(params, "Lx")
	mu = qom_mass_ratio(params)
	omega_pe = math.sqrt(mu)
	electron_skin_depth = 1.0 / omega_pe
	b0 = first_float(params, "B0x")
	epsilon = first_float(params, "WhistlerSeedEpsilon", 0.0)
	delta_b = epsilon * b0
	seed_mode = whistler_seed_mode(params)
	light_speed = first_float(params, "c", 1.0)
	k_di = TWO_PI * mode_x / lx
	k_de = k_di * electron_skin_depth
	omega_approx = cold_whistler_approx_frequency(mu, b0, k_di, light_speed)
	omega_exact = cold_whistler_exact_frequency(mu, b0, k_di, light_speed)
	use_exact = first_bool(params, "WhistlerUseExactColdRoot", True)
	omega_theory = omega_exact if use_exact else omega_approx
	dispersion_mode_max = args.max_dispersion_mode or max(args.max_spectrum_mode, mode_x)
	_, dispersion_k_de, dispersion_omega_approx, dispersion_omega_exact = whistler_dispersion_curve(
		dispersion_mode_max,
		lx,
		mu,
		b0,
		light_speed,
	)
	period = TWO_PI / omega_theory
	phase_speed_theory = omega_theory / k_di
	e_over_b = omega_theory / (light_speed * k_di)
	current_factor = light_speed * k_di - omega_theory * omega_theory / (light_speed * k_di)
	output_current_factor = current_factor / (4.0 * math.pi)

	b_path = data_dir / "B"
	e_path = data_dir / "E"
	j_path = data_dir / "J"
	time_pi, b_modes, first_spectrum, last_spectrum = read_b_right_history(
		b_path,
		mode_x,
		args.max_spectrum_mode,
		periodic,
	)

	fit_start = args.fit_start if args.fit_start is not None else time_pi[0]
	fit_end = args.fit_end if args.fit_end is not None else time_pi[-1]
	omega_pic, phase_intercept, gamma_pic, log_amp_intercept, phase = fit_phase_and_growth(
		time_pi,
		b_modes["b_right"],
		fit_start,
		fit_end,
	)

	e_modes = None
	j_modes = None
	if field_data_exists(data_dir, "E"):
		_, e_modes = read_component_modes(e_path, ("Ey", "Ez"), mode_x, periodic)
	if field_data_exists(data_dir, "J"):
		_, j_modes = read_component_modes(j_path, ("Jy", "Jz"), mode_x, periodic)

	output_path = args.output or (args.run_dir / "whistler_analysis.png")
	plot_analysis(
		output_path,
		time_pi,
		b_modes["b_right"],
		phase,
		first_spectrum,
		last_spectrum,
		mode_x,
		delta_b,
		k_de,
		omega_theory,
		omega_pic,
		gamma_pic,
		dispersion_k_de,
		dispersion_omega_approx,
		dispersion_omega_exact,
		phase_intercept,
		log_amp_intercept,
		fit_start,
		fit_end,
	)

	by_bz_amp_ratio = safe_median_ratio(b_modes["By"], b_modes["Bz"])
	phase_z_minus_y = float(np.median(wrap_to_pi(np.angle(b_modes["Bz"]) - np.angle(b_modes["By"]))))
	left_right_ratio = safe_median_ratio(b_modes["b_left"], b_modes["b_right"])
	first_mode_fraction = mode_power_fraction(first_spectrum, mode_x)
	last_mode_fraction = mode_power_fraction(last_spectrum, mode_x)
	output_dt = float(np.median(np.diff(time_pi))) if time_pi.size > 1 else float("nan")
	samples_per_period = period / output_dt if output_dt > 0.0 else float("nan")
	omega_error = omega_pic - omega_theory
	relative_omega_error = omega_error / omega_theory
	phase_speed_pic = omega_pic / k_di

	keep = (time_pi >= fit_start) & (time_pi <= fit_end)
	mode_energy = np.abs(b_modes["b_right"]) ** 2
	energy_variation = float((mode_energy[keep].max() - mode_energy[keep].min()) / mode_energy[keep].mean())

	print(f"input file              = {input_path}")
	print(f"data directory          = {data_dir}")
	print(f"seeded mode m           = {mode_x}")
	print(f"k d_i                   = {k_di:.6f}")
	print(f"k d_e                   = {k_de:.6f}")
	print(f"mu                      = {mu:.6f}")
	print(f"B0                      = {b0:.6f}")
	print(f"deltaB                  = {delta_b:.6e}")
	print(f"seed mode               = {seed_mode}")
	print(f"omega approx            = {omega_approx:.12f}")
	print(f"omega exact             = {omega_exact:.12f}")
	omega_label = "omega theory used" if seed_mode == "eigenmode" else "omega theory reference"
	print(f"{omega_label:<24}= {omega_theory:.12f}")
	print(f"omega PIC fit           = {omega_pic:.12f}")
	print(f"omega PIC-theory        = {omega_error:.6e}")
	print(f"relative omega error    = {relative_omega_error:.6e}")
	print(f"phase speed theory      = {phase_speed_theory:.12f}")
	print(f"phase speed PIC         = {phase_speed_pic:.12f}")
	print(f"dispersion modes shown  = 1..{dispersion_mode_max}")
	print(f"dispersion PIC point    = (k d_e={k_de:.6f}, omega={omega_pic:.12f})")
	print(f"period theory           = {period:.6f}")
	print(f"output dt               = {output_dt:.6f}")
	print(f"outputs per period      = {samples_per_period:.3f}")
	print(f"fit window omega_pi t   = [{fit_start:.6f}, {fit_end:.6f}]")
	print(f"PIC gamma/omega_pi      = {gamma_pic:.6e}")
	print(f"energy variation fit    = {energy_variation:.6e}")
	print(f"initial |b_R|/deltaB    = {abs(b_modes['b_right'][0]) / delta_b:.6f}")
	print(f"median |By|/|Bz|        = {by_bz_amp_ratio:.6f}")
	print(f"median phase Bz-By      = {phase_z_minus_y:.6f} rad")
	print(f"median |b_L|/|b_R|      = {left_right_ratio:.6e}")
	print(f"first seeded power frac = {first_mode_fraction:.6f}")
	print(f"last seeded power frac  = {last_mode_fraction:.6f}")
	print(f"theory |E|/|B|          = {e_over_b:.6f}")
	if e_modes is not None:
		ey_bz = safe_median_ratio(e_modes["Ey"], b_modes["Bz"])
		ez_by = safe_median_ratio(e_modes["Ez"], b_modes["By"])
		print(f"median |Ey|/|Bz|        = {ey_bz:.6f}")
		print(f"median |Ez|/|By|        = {ez_by:.6f}")
	print(f"theory |J|/|B|          = {current_factor:.6f}")
	print(f"expected output |J|/|B| = {output_current_factor:.6f}")
	if j_modes is not None:
		jy_by = safe_median_ratio(j_modes["Jy"], b_modes["By"])
		jz_bz = safe_median_ratio(j_modes["Jz"], b_modes["Bz"])
		print(f"median |Jy|/|By|        = {jy_by:.6f}")
		print(f"median |Jz|/|Bz|        = {jz_bz:.6f}")
	if samples_per_period < 20.0:
		print("warning                 = fewer than 20 field outputs per period; frequency fits may be coarse")
	print(f"plot written to         = {output_path}")


if __name__ == "__main__":
	main()
