extends Benchmark

func run_numdot(n: int) -> NDArray:
	var is_prime := nd.ones(n, nd.DType.Bool)
	is_prime.set(false, nd.to(2))
	
	var p := 2
	while p * p <= n:
		if is_prime.get_bool(p):
			is_prime.set(false, nd.range(p * p, null, p))
		p += 1
	return is_prime

# From https://www.geeksforgeeks.org/python-program-for-sieve-of-eratosthenes/
func run_gdscript(n: int) -> PackedByteArray:
	var is_prime := PackedByteArray()
	is_prime.resize(n)
	
	is_prime.fill(1)
	is_prime[0] = false
	is_prime[1] = false
	
	var p := 2
	while p * p <= n:
		if is_prime[p] == 1:
			# Updating all multiples of p
			for i in range(p * p, n, p):
				is_prime[i] = 0
		p += 1
	return is_prime

func run_benchmark() -> void:
	# Same test as https://www.youtube.com/watch?v=qDXomV7Ojko
	var n := 2_000_000

	print("Sieve of Eratosthenes with n=" + str(n))
	
	# Examples from my computer:
	# n=200            25 GDScript
	#                  50 NumDot
	# n=2000          100 GDScript
	#                  60 NumDot
	# n=20000        1100 GDScript
	#                 130 NumDot
	# n=200000      13000 GDScript
	#                 780 NumDot
	# n=2000000    150000 GDScript
	#                9300 NumDot
	# n=20000000  1600000 GDScript
	#              110000 NumDot

	begin_section("GDScript")
	var result_gd := run_gdscript(n)
	store_result()

	begin_section("NumDot")
	var result_nd := run_numdot(n)
	store_result()

	assert(result_gd == result_nd.to_packed_byte_array())
	end()
