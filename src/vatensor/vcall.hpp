#ifndef VCALL_HPP
#define VCALL_HPP

#include "create.hpp"
#include "varray.hpp"
#include "vassign.hpp"
#include "vfunc/ufunc.hpp"

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
	void call_vfunc_unary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableUnary& table, const VArrayTarget& target, const VData& a, Args&&... args) {
		const DType a_type = va::dtype(a);

		const auto& ufunc = table[a_type];
		if (ufunc.function_ptr == nullptr) throw std::runtime_error("Unsupported dtype for ufunc.");

		std::shared_ptr<VArray> temp(nullptr);
		auto& target_ = evaluate_target(allocator, target, ufunc.output_dtype, va::shape(a), temp);

		if (a_type == ufunc.input_types[0]) {
			reinterpret_cast<_call::UnaryDummyFunction<Args...>>(ufunc.function_ptr)(
				*static_cast<_call::Dummy*>(_call::get_value_ptr(target_)),
				*static_cast<const _call::Dummy*>(_call::get_value_ptr(a)),
				std::forward<Args...>(args)...
			);
		}
		else {
			const auto a_ = _call::_copy_as_dtype(allocator, a, ufunc.input_types[0]);
			reinterpret_cast<_call::UnaryDummyFunction<Args...>>(ufunc.function_ptr)(
				*static_cast<_call::Dummy*>(_call::get_value_ptr(target_)),
				*static_cast<const _call::Dummy*>(_call::get_value_ptr(a_->data)),
				std::forward<Args...>(args)...
			);
		}

		if (temp != nullptr) {
			// We wrote to temp because the return type mismatched; now we need to resolve that.
			va::assign(*std::get<va::VData*>(target), temp->data);
		}
	}

	template<typename A, typename B, typename... Args>
	void call_vfunc_binary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const shape_type& result_shape, const A& a, const B& b, Args&&... args) {
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
					std::forward<Args...>(args)...
				);
				break;
			}
			case 0b10: {
				const auto b_ = _call::_copy_as_dtype(allocator, b, ufunc.input_types[1]);
				reinterpret_cast<_call::BinaryDummyFunction<Args...>>(ufunc.function_ptr)(
					*static_cast<_call::Dummy*>(_call::get_value_ptr(target_)),
					*static_cast<const _call::Dummy*>(_call::get_value_ptr(a)),
					*static_cast<const _call::Dummy*>(_call::get_value_ptr(_call::deref(b_))),
					std::forward<Args...>(args)...
				);
				break;
			}
			case 0b01: {
				const auto a_ = _call::_copy_as_dtype(allocator, a, ufunc.input_types[0]);
				reinterpret_cast<_call::BinaryDummyFunction<Args...>>(ufunc.function_ptr)(
					*static_cast<_call::Dummy*>(_call::get_value_ptr(target_)),
					*static_cast<const _call::Dummy*>(_call::get_value_ptr(_call::deref(a_))),
					*static_cast<const _call::Dummy*>(_call::get_value_ptr(b)),
					std::forward<Args...>(args)...
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
					std::forward<Args...>(args)...
				);
				break;
			}
		}

		if (temp != nullptr) {
			// We wrote to temp because the return type mismatched; now we need to resolve that.
			va::assign(*std::get<VData*>(target), temp->data);
		}
	}

	void call_ufunc_unary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableUnary& table, const VArrayTarget& target, const VData& a);
	void call_ufunc_binary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const VData& a, const VData& b);

	void call_ufunc_binary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const VScalar& a, const VData& b);
	void call_ufunc_binary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const VData& a, const VScalar& b);
}

#endif //VCALL_HPP
