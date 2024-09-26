extends Node

func numdot(n: int) -> NDArray:
	var is_prime := nd.ones(n, nd.DType.Bool)
	is_prime.set(false, nd.to(2))
	
	var p := 2
	while p * p <= n:
		if is_prime.get_bool(p):
			is_prime.set(false, nd.range(p * p, null, p))
		p += 1
	return is_prime

# From https://www.geeksforgeeks.org/python-program-for-sieve-of-eratosthenes/
func gdscript(n: int) -> PackedByteArray:
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

func _ready() -> void:
	# Same test as https://www.youtube.com/watch?v=qDXomV7Ojko
	# n=2,000,000
	var n := 2000000
	var start_time: int
	print("Sieve of Eratosthenes with n=" + str(n))
	
	# On my computer, this takes about 150,000us
	start_time = Time.get_ticks_usec()
	var result_gd := gdscript(n)
	print("GDScript: " + str(Time.get_ticks_usec() - start_time))

	# On my computer, this takes about 75,000us
	start_time = Time.get_ticks_usec()
	var result_nd := numdot(n)
	print("NumDot: " + str(Time.get_ticks_usec() - start_time))

	assert(result_gd == result_nd.to_packed_byte_array())
