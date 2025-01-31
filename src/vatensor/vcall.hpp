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

	VData& evaluate_target(VStoreAllocator& allocator, const VArrayTarget& target, DType dtype, const shape_type& result_shape, std::shared_ptr<VArray>& temp);

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

	template<typename... Args>
	void call_rfunc_unary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableUnary& table, const VArrayTarget& target, const VData& a, const va::axes_type* axes, Args&&... args) {
		if (axes) {
			va::shape_type result_shape = va::shape(a);
			bool mask[result_shape.size()];
			std::fill_n(mask, result_shape.size(), true);

			for (const auto axis : *axes)
			{
				const size_t axis_normal = va::util::normalize_axis(axis, result_shape.size());
				if (!mask[axis_normal]) {
					throw std::runtime_error("Duplicate value in 'axis'.");
				}
				mask[axis_normal] = false;
			}

			result_shape.erase(
				std::remove_if(
					result_shape.begin(),
					result_shape.end(),
					[&mask, index = 0](int) mutable { return !mask[index++]; }
				),
				result_shape.end()
			);

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
		const auto& a_shape = va::shape(a);
		const auto& b_shape = va::shape(b);

		auto result_shape = shape_type(std::max(a_shape.size(), b_shape.size()));
		std::fill_n(result_shape.begin(), result_shape.size(), std::numeric_limits<shape_type::value_type>::max());
		xt::broadcast_shape(a_shape, result_shape);
		xt::broadcast_shape(b_shape, result_shape);

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

	static VData &unwrap_target(const va::VArrayTarget& target) {
		if (const auto target_data = std::get_if<VData*>(&target)) {
			return **target_data;
		}
		return (*std::get<std::shared_ptr<VArray>*>(target))->data;
	}
}

#endif //VCALL_HPP
