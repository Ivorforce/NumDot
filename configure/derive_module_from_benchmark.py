import dataclasses
import json
import pathlib

import numpy as np
import pandas as pd

def main():
	with pathlib.Path("ufuncs.json").open("r") as f:
		ufuncs_json = json.load(f)

	# To use this:
	# - Compile the project with the 'previous' optimization as the base.
	# - Rename the benchmark.csv file to benchmark-base.csv.
	# - Compile the project with the target optimization.
	# - Run this module.
	benchmark_base = pd.read_csv("../tests/gen/benchmark-base.csv", index_col=0)["numdot"]
	benchmark_module = pd.read_csv("../tests/gen/benchmark.csv", index_col=0)["numdot"]

	assert set(benchmark_base.index) == set(benchmark_module.index), "Incompatible benchmarks, re-run tests."

	better_ufuncs = []
	for ufunc in sorted(ufuncs_json["ufuncs"], key=lambda d: d["ufunc"]):
		ufunc_name = ufunc["ufunc"]
		relevant_tests = [
			test_name
			for test_name in benchmark_base.index
			if test_name.startswith(f"{ufunc_name}_")
		]

		if not relevant_tests:
			print(f"Warning: ufunc not benchmarked, cannot be optimized: {ufunc_name}")
			continue

		# Lower is better
		diffs = benchmark_module[relevant_tests] / benchmark_base[relevant_tests]
		median_diff = np.quantile(diffs, 0.2)
		if median_diff > 1.1:
			print(f"Warning: ufunc got worse somehow: {ufunc_name}")
			continue
		elif median_diff > 0.95:
			continue  # No large difference

		print(f"ufunc got better: {ufunc_name} ({median_diff:.2f})")
		better_ufuncs.append(ufunc_name)

	print("Better ufuncs: ", json.dumps(better_ufuncs))

if __name__ == "__main__":
	main()
