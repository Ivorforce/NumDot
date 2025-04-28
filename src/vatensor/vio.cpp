#include "vio.hpp"

#include "xtensor/io/xnpy.hpp"
#include "array_store.hpp"

struct membuf : std::streambuf {
	membuf(char* begin, char* end) {
		this->setg(begin, begin, end);
	}

	pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which = std::ios_base::in) override
	{
		if (dir == std::ios_base::cur)
			gbump(off);
		else if (dir == std::ios_base::end)
			setg(eback(), egptr() + off, egptr());
		else if (dir == std::ios_base::beg)
			setg(eback(), eback() + off, egptr());
		return gptr() - eback();
	}

	pos_type seekpos(pos_type sp, std::ios_base::openmode which) override
	{
		return seekoff(sp - pos_type(off_type(0)), std::ios_base::beg, which);
	}
};

va::DType dtype_from_typestring(std::string& type_string) {
	// First char is endianness.
	// TODO Is there no better way to do this??
	std::string str = type_string.substr(1);
	if (str == "b1") return va::DType::Bool;
	else if (str == "u1") return va::DType::UInt8;
	else if (str == "u2") return va::DType::UInt16;
	else if (str == "u4") return va::DType::UInt32;
	else if (str == "u8") return va::DType::UInt64;
	else if (str == "i1") return va::DType::Int8;
	else if (str == "i2") return va::DType::Int16;
	else if (str == "i4") return va::DType::Int32;
	else if (str == "i8") return va::DType::Int64;
	else if (str == "f4") return va::DType::Float32;
	else if (str == "f8") return va::DType::Float64;
	else if (str == "c8") return va::DType::Complex64;
	else if (str == "c16") return va::DType::Complex128;
	else return va::DTypeMax;
}

std::shared_ptr<va::VArray> va::load_npy(const char* data, std::size_t size) {
	auto sbuf = membuf(const_cast<char*>(data), const_cast<char*>(data) + size);
	std::istream in(&sbuf);

	xt::detail::npy_file file = xt::detail::load_npy_file(in);
	DType dtype = dtype_from_typestring(file.m_typestring);
	if (dtype == va::DTypeMax) {
		throw std::runtime_error("Npy type is not supported");
	}

	std::vector<std::size_t> strides(file.m_shape.size());

	// check if the typestring matches the given one
	std::visit([&file](auto t) {
		using T = decltype(t);
		if (file.m_typestring != xt::detail::build_typestring<T>())
			throw std::runtime_error("Npy type is not supported");
	}, dtype_to_variant(dtype));

	// Move over ownership.
	store::VCharPtrStore store { file.m_buffer, file.m_n_bytes, dtype };
	file.m_buffer = nullptr;

	return va::store::from_store(
		std::move(store),
		file.m_shape,
		strides_type {}, // ignored
		file.m_fortran_order ? xt::layout_type::column_major : xt::layout_type::row_major
	);
}

std::string va::save_npy(VData& data) {
	return std::visit([](const auto& data) {
		return xt::dump_npy(data);
	}, data);
}
