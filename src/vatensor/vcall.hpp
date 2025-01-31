#ifndef VCALL_HPP
#define VCALL_HPP

#include "create.hpp"
#include "varray.hpp"
#include "vassign.hpp"
#include "vfunc/tables.hpp"

namespace va {
	namespace _call {
		using Dummy = char;
		template<typename... Args>
		using InplaceDummyFunction = void (*)(Dummy& b, Args... args);
		template<typename... Args>
		using UnaryDummyFunction = void (*)(Dummy& a, const Dummy& b, Args... args);
		template<typename... Args>
		using BinaryDummyFunction = void (*)(Dummy& a, const Dummy& b, const Dummy& c, Args... args);

		inline std::shared_ptr<VArray> _copy_as_dtype(VStoreAllocator& allocator, const VData& a, DType dtype) {
			return va::copy_as_dtype(allocator, a, dtype);
		}

		inline VScalar _copy_as_dtype(VStoreAllocator& allocator, const VScalar& a, DType dtype) {
			return va::static_cast_scalar(a, dtype);
		}

		inline const VScalar& deref(const VScalar& a) {
			return a;
		}

		inline const VData& deref(const std::shared_ptr<VArray>& a) {
			return a->data;
		}

		template<typename... Args>
		void* get_value_ptr(std::variant<Args...>& variant) {
			return std::visit([](auto& arg) -> void* { return &arg; }, variant);
		}

		template<typename... Args>
		const void* get_value_ptr(const std::variant<Args...>& variant) {
			return std::visit([](auto& arg) -> const void* { return &arg; }, variant);
		}
	}

	void fill(VData& a, const VScalar& fill_value);
	void assign(VData& a, const VData& b);

	VData& evaluate_target(VStoreAllocator& allocator, const VArrayTarget& target, DType dtype, const shape_type& result_shape, std::shared_ptr<VArray>& temp);

	template<typename... Args>
	void _call_vfunc_inplace(const vfunc::tables::UFuncTableInplace& table, VData& a, Args&&... args) {
		const DType a_type = va::dtype(a);

		const auto& function_ptr = table[a_type];
		if (function_ptr == nullptr) throw std::runtime_error("Unsupported dtype for ufunc.");

		reinterpret_cast<_call::InplaceDummyFunction<Args...>>(function_ptr)(
			*static_cast<_call::Dummy*>(_call::get_value_ptr(a)),
			std::forward<Args>(args)...
		);
	}

	template<typename... Args>
	void _call_vfunc_inplace_binary(const vfunc::tables::UFuncTableInplaceBinary& table, VData& a, const VData& b, Args&&... args) {
		const DType a_type = va::dtype(a);
		const DType b_type = va::dtype(b);

		const auto& function_ptr = table[a_type][b_type];
		if (function_ptr == nullptr) throw std::runtime_error("Unsupported dtype for ufunc.");

		reinterpret_cast<_call::UnaryDummyFunction<Args...>>(function_ptr)(
			*static_cast<_call::Dummy*>(_call::get_value_ptr(a)),
			*static_cast<const _call::Dummy*>(_call::get_value_ptr(b)),
			std::forward<Args>(args)...
		);
	}

	template<typename... Args>
	void _call_vfunc_unary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableUnary& table, const VArrayTarget& target, const shape_type& result_shape, const VData& a, Args&&... args) {
		const DType a_type = va::dtype(a);

		const auto& ufunc = table[a_type];
		if (ufunc.function_ptr == nullptr) throw std::runtime_error("Unsupported dtype for ufunc.");

		std::shared_ptr<VArray> temp(nullptr);
		auto& target_ = evaluate_target(allocator, target, ufunc.output_dtype, result_shape, temp);

		if (a_type == ufunc.input_types[0]) {
			reinterpret_cast<_call::UnaryDummyFunction<Args...>>(ufunc.function_ptr)(
				*static_cast<_call::Dummy*>(_call::get_value_ptr(target_)),
				*static_cast<const _call::Dummy*>(_call::get_value_ptr(a)),
				std::forward<Args>(args)...
			);
		}
		else {
			const auto a_ = _call::_copy_as_dtype(allocator, a, ufunc.input_types[0]);
			reinterpret_cast<_call::UnaryDummyFunction<Args...>>(ufunc.function_ptr)(
				*static_cast<_call::Dummy*>(_call::get_value_ptr(target_)),
				*static_cast<const _call::Dummy*>(_call::get_value_ptr(a_->data)),
				std::forward<Args>(args)...
			);
		}

		if (temp != nullptr) {
			// We wrote to temp because the return type mismatched; now we need to resolve that.
			va::assign(*std::get<va::VData*>(target), temp->data);
		}
	}

	template<typename... Args>
	void call_vfunc_unary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableUnary& table, const VArrayTarget& target, const VData& a, Args&&... args) {
		_call_vfunc_unary(allocator, table, target, va::shape(a), a, std::forward<Args>(args)...);
	}

	void shape_reduce_axes(va::shape_type& shape, const va::axes_type& axes);
	va::shape_type combined_shape(const shape_type& a_shape, const shape_type& b_shape);

	template<typename... Args>
	void call_rfunc_unary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableUnary& table, const VArrayTarget& target, const VData& a, const va::axes_type* axes, Args&&... args) {
		if (axes) {
			va::shape_type result_shape = va::shape(a);
			shape_reduce_axes(result_shape, *axes);

			_call_vfunc_unary(allocator, table, target, result_shape, a, std::move(axes), std::forward<Args>(args)...);
		}
		else {
			_call_vfunc_unary(allocator, table, target, va::shape_type(), a, nullptr, std::forward<Args>(args)...);
		}
	}

	template<typename A, typename B, typename... Args>
	void _call_vfunc_binary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const shape_type& result_shape, const A& a, const B& b, Args&&... args) {
		const DType a_type = va::dtype(a);
		const DType b_type = va::dtype(b);

		const auto& ufunc = table[a_type][b_type];
		if (ufunc.function_ptr == nullptr) throw std::runtime_error("Unsupported dtype for ufunc.");

		std::shared_ptr<VArray> temp(nullptr);
		auto& target_ = evaluate_target(allocator, target, ufunc.output_dtype, result_shape, temp);

		switch ((static_cast<uint8_t>(a_type == ufunc.input_types[0]) << 1) | static_cast<uint8_t>(b_type == ufunc.input_types[1])) {
			case 0b11: {
				reinterpret_cast<_call::BinaryDummyFunction<Args...>>(ufunc.function_ptr)(
					*static_cast<_call::Dummy*>(_call::get_value_ptr(target_)),
					*static_cast<const _call::Dummy*>(_call::get_value_ptr(a)),
					*static_cast<const _call::Dummy*>(_call::get_value_ptr(b)),
					std::forward<Args>(args)...
				);
				break;
			}
			case 0b10: {
				const auto b_ = _call::_copy_as_dtype(allocator, b, ufunc.input_types[1]);
				reinterpret_cast<_call::BinaryDummyFunction<Args...>>(ufunc.function_ptr)(
					*static_cast<_call::Dummy*>(_call::get_value_ptr(target_)),
					*static_cast<const _call::Dummy*>(_call::get_value_ptr(a)),
					*static_cast<const _call::Dummy*>(_call::get_value_ptr(_call::deref(b_))),
					std::forward<Args>(args)...
				);
				break;
			}
			case 0b01: {
				const auto a_ = _call::_copy_as_dtype(allocator, a, ufunc.input_types[0]);
				reinterpret_cast<_call::BinaryDummyFunction<Args...>>(ufunc.function_ptr)(
					*static_cast<_call::Dummy*>(_call::get_value_ptr(target_)),
					*static_cast<const _call::Dummy*>(_call::get_value_ptr(_call::deref(a_))),
					*static_cast<const _call::Dummy*>(_call::get_value_ptr(b)),
					std::forward<Args>(args)...
				);
				break;
			}
			case 0b00: {
				const auto a_ = _call::_copy_as_dtype(allocator, a, ufunc.input_types[0]);
				const auto b_ = _call::_copy_as_dtype(allocator, b, ufunc.input_types[1]);
				reinterpret_cast<_call::BinaryDummyFunction<Args...>>(ufunc.function_ptr)(
					*static_cast<_call::Dummy*>(_call::get_value_ptr(target_)),
					*static_cast<const _call::Dummy*>(_call::get_value_ptr(_call::deref(a_))),
					*static_cast<const _call::Dummy*>(_call::get_value_ptr(_call::deref(b_))),
					std::forward<Args>(args)...
				);
				break;
			}
		}

		if (temp != nullptr) {
			// We wrote to temp because the return type mismatched; now we need to resolve that.
			va::assign(*std::get<VData*>(target), temp->data);
		}
	}

	template<typename... Args>
	void call_vfunc_binary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const VData& a, const VData& b, Args&&... args) {
		const shape_type result_shape = combined_shape(va::shape(a), va::shape(b));
		_call_vfunc_binary(allocator, table, target, result_shape, a, b, std::forward<Args>(args)...);
	}

	template<typename... Args>
	void call_vfunc_binary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const VScalar& a, const VData& b, Args&&... args) {
		_call_vfunc_binary(allocator, table, target, va::shape(b), a, b, std::forward<Args>(args)...);
	}

	template<typename... Args>
	void call_vfunc_binary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const VData& a, const VScalar& b, Args&&... args) {
		_call_vfunc_binary(allocator, table, target, va::shape(a), a, b, std::forward<Args>(args)...);
	}

	template <typename... Args>
	inline void call_vfunc_binary(va::VStoreAllocator& allocator, const va::vfunc::tables::UFuncTablesBinaryCommutative& table, const va::VArrayTarget& target, const va::VData& a, const va::VData& b, Args... args) {
		if (va::dimension(a) == 0) return va::call_vfunc_binary(allocator, table.scalar_right, target, b, va::to_single_value(a), args...);
		if (va::dimension(b) == 0) return va::call_vfunc_binary(allocator, table.scalar_right, target, a, va::to_single_value(b), args...);
		va::call_vfunc_binary(allocator, table.tensors, target, a, b, args...);
	}

	template <typename... Args>
	inline void call_vfunc_binary(va::VStoreAllocator& allocator, const va::vfunc::tables::UFuncTablesBinary& table, const va::VArrayTarget& target, const va::VData& a, const va::VData& b, Args... args) {
		if (va::dimension(a) == 0) return va::call_vfunc_binary(allocator, table.scalar_left, target, va::to_single_value(a), b, args...);
		if (va::dimension(b) == 0) return va::call_vfunc_binary(allocator, table.scalar_right, target, a, va::to_single_value(b), args...);
		va::call_vfunc_binary(allocator, table.tensors, target, a, b, args...);
	}

	template<typename... Args>
	void call_rfunc_binary(VStoreAllocator& allocator, const vfunc::tables::UFuncTablesBinaryCommutative& table, const VArrayTarget& target, const VData& a, const VData& b, Args&&... args) {
		if (va::dimension(a) == 0) return _call_vfunc_binary(allocator, table.scalar_right, target, va::shape_type(), b, va::to_single_value(a), args...);
		if (va::dimension(b) == 0) return _call_vfunc_binary(allocator, table.scalar_right, target, va::shape_type(), a, va::to_single_value(b), args...);
		_call_vfunc_binary(allocator, table.tensors, target, va::shape_type(), a, b, std::forward<Args>(args)...);
	}

	template<typename... Args>
	void call_rfunc_binary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const VData& a, const VData& b, const va::axes_type* axes, Args&&... args) {
		if (axes) {
			shape_type result_shape = combined_shape(va::shape(a), va::shape(b));
			shape_reduce_axes(result_shape, *axes);

			_call_vfunc_binary(allocator, table, target, result_shape, a, b, std::move(axes), std::forward<Args>(args)...);
		}
		else {
			_call_vfunc_binary(allocator, table, target, va::shape_type(), a, b, nullptr, std::forward<Args>(args)...);
		}
	}
}

#endif //VCALL_HPP
