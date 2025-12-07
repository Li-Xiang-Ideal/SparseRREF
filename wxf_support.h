/*
	Copyright (C) 2025 Zhenjie Li (Li, Zhenjie)

	You can redistribute it and/or modify it under the terms of the MIT
	License.
*/

/*
	WXF is a binary format for faithfully serializing Wolfram Language expressions
	in a form suitable for outside storage or interchange with other programs.
	WXF can readily be interpreted using low-level native types available in many
	programming languages, making it suitable as a format for reading and writing
	Wolfram Language expressions in other programming languages.

	The details of the WXF format are described in the Wolfram Language documentation:
	https://reference.wolfram.com/language/tutorial/WXFFormatDescription.html.en .

	The full list of supported types is given below:

	done	byte value  type of part
	*		102			function
	*		67			int8_t
	*		106			int16_t
	*		105			int32_t
	*		76			int64_t
	*		114			machine reals
	*		83			string
	*		66			binary string
	*		115			symbol
	*		73			big integer
	*		82			big real
	*		193			packed array
	*		194			numeric array
	*		65			association
	*		58			delayed rule in association
	*		45			rule in association
*/

/*
	an example to test:
		SparseArray[{{1, 1} -> 1/3.0, {1, 23133} ->
		N[Pi, 100] + I N[E, 100], {44, 2} -> -(4/
		 33333333333333444333333335), {_, _} -> 0}]

	FullForm:
		SparseArray[Automatic,List[44,23133],0,
		List[1,List[List[0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
		2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3],
		List[List[1],List[23133],List[2]]],
		List[0.3333333333333333`,
		Complex[3.1415926535897932384626433832795028841971693993751058209
		749445923078164062862089986280348253421170679821480865191976`100.,
		2.7182818284590452353602874713526624977572470936999595749669676277
		240766303535475945713821785251664274274663919320031`100.],
		Rational[-4,33333333333333444333333335]]]]

	std::vector<uint8_t> test{ 56, 58, 102, 4, 115, 11, 83, 112, 97, 114, 115, 101, 65, 114, 114, \
							97, 121, 115, 9, 65, 117, 116, 111, 109, 97, 116, 105, 99, 193, 1, 1, \
							2, 44, 0, 93, 90, 67, 0, 102, 3, 115, 4, 76, 105, 115, 116, 67, 1, \
							102, 2, 115, 4, 76, 105, 115, 116, 193, 0, 1, 45, 0, 2, 2, 2, 2, 2, \
							2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, \
							2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 193, 1, 2, 3, 1, 1, \
							0, 93, 90, 2, 0, 102, 3, 115, 4, 76, 105, 115, 116, 114, 85, 85, 85, \
							85, 85, 85, 213, 63, 102, 2, 115, 7, 67, 111, 109, 112, 108, 101, \
							120, 82, 122, 51, 46, 49, 52, 49, 53, 57, 50, 54, 53, 51, 53, 56, 57, \
							55, 57, 51, 50, 51, 56, 52, 54, 50, 54, 52, 51, 51, 56, 51, 50, 55, \
							57, 53, 48, 50, 56, 56, 52, 49, 57, 55, 49, 54, 57, 51, 57, 57, 51, \
							55, 53, 49, 48, 53, 56, 50, 48, 57, 55, 52, 57, 52, 52, 53, 57, 50, \
							51, 48, 55, 56, 49, 54, 52, 48, 54, 50, 56, 54, 50, 48, 56, 57, 57, \
							56, 54, 50, 56, 48, 51, 52, 56, 50, 53, 51, 52, 50, 49, 49, 55, 48, \
							54, 55, 57, 56, 50, 49, 52, 56, 48, 56, 54, 53, 49, 57, 49, 57, 55, \
							54, 96, 49, 48, 48, 46, 82, 122, 50, 46, 55, 49, 56, 50, 56, 49, 56, \
							50, 56, 52, 53, 57, 48, 52, 53, 50, 51, 53, 51, 54, 48, 50, 56, 55, \
							52, 55, 49, 51, 53, 50, 54, 54, 50, 52, 57, 55, 55, 53, 55, 50, 52, \
							55, 48, 57, 51, 54, 57, 57, 57, 53, 57, 53, 55, 52, 57, 54, 54, 57, \
							54, 55, 54, 50, 55, 55, 50, 52, 48, 55, 54, 54, 51, 48, 51, 53, 51, \
							53, 52, 55, 53, 57, 52, 53, 55, 49, 51, 56, 50, 49, 55, 56, 53, 50, \
							53, 49, 54, 54, 52, 50, 55, 52, 50, 55, 52, 54, 54, 51, 57, 49, 57, \
							51, 50, 48, 48, 51, 49, 96, 49, 48, 48, 46, 102, 2, 115, 8, 82, 97, \
							116, 105, 111, 110, 97, 108, 67, 252, 73, 26, 51, 51, 51, 51, 51, 51, \
							51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 51, 51, 51, 51, 51, 51, \
							51, 51, 53 };

	auto tree = WXF_PARSER::MakeExprTree(test);
	for (const auto& token : tree.tokens) {
		token.print();
	};

	result:
		func: 4 vars
		symbol: SparseArray
		symbol: Automatic
		array: rank = 1, dimensions = 2
		data: 44 23133
		i8: 0
		func: 3 vars
		symbol: List
		i8: 1
		func: 2 vars
		symbol: List
		array: rank = 1, dimensions = 45
		data: 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3
		array: rank = 2, dimensions = 3 1
		data: 1 23133 2
		func: 3 vars
		symbol: List
		f64: 0.333333
		func: 2 vars
		symbol: Complex
		bigreal: 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865191976`100.
		bigreal: 2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274274663919320031`100.
		func: 2 vars
		symbol: Rational
		i8: -4
		bigint: 33333333333333444333333335
*/

#pragma once

#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <functional>
#include <vector>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>

namespace WXF_PARSER {

	using complex_float_t = std::complex<float>;
	using complex_double_t = std::complex<double>;

	enum class WXF_HEAD {
		// function type
		func = 102,
		association = 65,
		delay_rule = 58,
		rule = 45,
		// string type
		symbol = 115,
		string = 83,
		binary_string = 66,
		bigint = 73,
		bigreal = 82,
		// number type
		i8 = 67,
		i16 = 106,
		i32 = 105,
		i64 = 76,
		f64 = 114,
		// array type
		array = 193,
		narray = 194
	};

	size_t size_of_head_num_type(const WXF_HEAD head) {
		switch (head) {
		case WXF_HEAD::i8:
			return sizeof(int8_t);
		case WXF_HEAD::i16:
			return sizeof(int16_t);
		case WXF_HEAD::i32:
			return sizeof(int32_t);
		case WXF_HEAD::i64:
			return sizeof(int64_t);
		case WXF_HEAD::f64:
			return sizeof(double);
		default:
			return 0;
		}
	}

	// array: head(numeric/packed array), num_type, rank, dimensions, data
	// for the num_type
	// 0 is int8_t      1 is int16_t
	// 2 is int32_t     3 is int64_t
	// 16 is uint8_t    17 is uint16_t ; only for numeric array
	// 18 is uint32_t   19 is uint64_t ; only for numeric array
	// 34 float         35 double
	// 51 complex float 52 complex double

	// only the last 3 bits are used to indicate the size
	size_t size_of_arr_num_type(const int num_type) {
		return size_t(1) << (num_type & 0b111);
	}

	template <typename T>
		requires std::is_integral_v<T>&& std::is_signed_v<T>
	constexpr uint8_t minimal_signed_bits(T x) noexcept {
		if (x >= INT8_MIN && x <= INT8_MAX) return 0;
		if (x >= INT16_MIN && x <= INT16_MAX) return 1;
		if (x >= INT32_MIN && x <= INT32_MAX) return 2;
		return 3; // for int64_t
	}

	// for positive signed/unsigned integer
	template <typename T>
	constexpr uint8_t minimal_pos_signed_bits(T x) noexcept {
		if (x <= INT8_MAX) return 0;
		if (x <= INT16_MAX) return 1;
		if (x <= INT32_MAX) return 2;
		if (x <= INT64_MAX) return 3;
		return 4; // for uint64_t
	}

	template <typename T>
		requires std::is_integral_v<T>&& std::is_unsigned_v<T>
	constexpr uint8_t minimal_unsigned_bits(T x) noexcept {
		if (x <= UINT8_MAX) return 0;
		if (x <= UINT16_MAX) return 1;
		if (x <= UINT32_MAX) return 2;
		return 3; // for uint64_t
	}

	inline void serialize_varint(std::vector<uint8_t>& buffer, uint64_t val) {
		uint8_t temp[10];
		size_t i = 0;

		do {
			temp[i] = val & 0x7F;
			val >>= 7;
			if (val != 0) temp[i] |= 0x80;
			++i;
		} while (val != 0);

		buffer.insert(buffer.end(), temp, temp + i);
	}

	template <typename T>
	void serialize_binary(std::vector<uint8_t>& buffer, const T& value) {
		const size_t old_size = buffer.size();
		buffer.resize(old_size + sizeof(T));
		std::memcpy(buffer.data() + old_size, &value, sizeof(T));
	}

	template <typename T>
	void serialize_binary(std::vector<uint8_t>& buffer, const T* valptr, const size_t len) {
		const size_t old_size = buffer.size();
		buffer.resize(old_size + sizeof(T) * len);
		std::memcpy(buffer.data() + old_size, valptr, sizeof(T) * len);
	}

	struct Encoder {
		std::vector<uint8_t> buffer;


		Encoder() = default;
		~Encoder() = default;
		Encoder(const Encoder&) = default;
		Encoder& operator=(const Encoder&) = default;
		Encoder(Encoder&&) = default;
		Encoder& operator=(Encoder&&) = default;

		void clear() { buffer.clear(); }

		// move from existing buffer
		Encoder(std::vector<uint8_t>&& buf) : buffer(std::move(buf)) {}

		// push ustr directly
		Encoder& push_ustr(const std::span<uint8_t> str) { buffer.insert(buffer.end(), str.begin(), str.end()); return *this; }
		Encoder& push_ustr(const std::vector<uint8_t>& str) { buffer.insert(buffer.end(), str.begin(), str.end()); return *this; }
		Encoder& push_ustr(const std::string_view str) {
			buffer.insert(buffer.end(), (uint8_t*)str.data(), (uint8_t*)(str.data() + str.size())); return *this;
		}
		template<typename T>
		Encoder& push_ustr(const T* str_ptr, const size_t len) {
			buffer.insert(buffer.end(), (uint8_t*)str_ptr, (uint8_t*)(str_ptr + len)); return *this;
		}
		template<typename T>
		Encoder& push_ustr(T* start, T* end) {
			buffer.insert(buffer.end(), (uint8_t*)start, (uint8_t*)end); return *this;
		}

		Encoder& push_integer(const int64_t val) {
			auto num_type = minimal_signed_bits(val);
			switch (num_type) {
			case 0:
				buffer.push_back((uint8_t)WXF_HEAD::i8);
				serialize_binary(buffer, (int8_t)val);
				break;
			case 1:
				buffer.push_back((uint8_t)WXF_HEAD::i16);
				serialize_binary(buffer, (int16_t)val);
				break;
			case 2:
				buffer.push_back((uint8_t)WXF_HEAD::i32);
				serialize_binary(buffer, (int32_t)val);
				break;
			case 3:
				buffer.push_back((uint8_t)WXF_HEAD::i64);
				serialize_binary(buffer, val);
				break;
			default:
				break;
			}
			return *this;
		}

		Encoder& push_real(const double val) {
			buffer.push_back((uint8_t)WXF_HEAD::f64);
			serialize_binary(buffer, val);
			return *this;
		}

		// push string type struct: string/symbol/bigint/bigreal, default type is string
		Encoder& push_string(const std::string_view str, const WXF_HEAD type = WXF_HEAD::string) {
			buffer.push_back((uint8_t)type);
			serialize_varint(buffer, str.size());
			return push_ustr(str);
		}

		Encoder& push_symbol(const std::string_view sym) { return push_string(sym, WXF_HEAD::symbol); }
		Encoder& push_bigint(const std::string_view bigint_str) { return push_string(bigint_str, WXF_HEAD::bigint); }
		Encoder& push_bigreal(const std::string_view bigreal_str) { return push_string(bigreal_str, WXF_HEAD::bigreal); }
		Encoder& push_binary_string(const std::string_view bin_str) { return push_string(bin_str, WXF_HEAD::binary_string); }

		Encoder& push_function(const std::string_view head, const size_t num_vars) {
			buffer.push_back((uint8_t)WXF_HEAD::func);
			serialize_varint(buffer, num_vars);
			return push_string(head, WXF_HEAD::symbol);
		}

		Encoder& push_association(const size_t num_rules) {
			buffer.push_back((uint8_t)WXF_HEAD::association);
			serialize_varint(buffer, num_rules);
			return *this;
		}

		Encoder& push_rule() { buffer.push_back((uint8_t)WXF_HEAD::rule); return *this; }
		Encoder& push_delay_rule() { buffer.push_back((uint8_t)WXF_HEAD::delay_rule); return *this; }

		// return the total length of the array
		size_t push_array_info(const std::vector<size_t>& dimension_array, WXF_HEAD type, uint8_t num_type) {
			size_t all_len = 1;
			// [array_type, num_type, rank, dimensions...]
			buffer.push_back((uint8_t)type);
			buffer.push_back(num_type);
			serialize_varint(buffer, dimension_array.size());
			for (auto dim : dimension_array) {
				serialize_varint(buffer, dim);
				all_len *= dim;
			}
			return all_len;
		}

		template<typename T>
		Encoder& push_array(const std::vector<size_t>& dimension_array, const std::span<T> data, WXF_HEAD type, uint8_t num_type) {
			// backup current size
			size_t old_size = buffer.size();

			// [array_type, num_type, rank, dimensions..., data...]
			auto all_len = push_array_info(dimension_array, type, num_type);

			if (all_len != data.size()) {
				std::cerr << "Encoder::push_array: Data size does not match the dimension array." << std::endl;
				// restore buffer
				buffer.resize(old_size);
				return *this;
			}

			// push data
			return push_ustr(data.data(), data.size());
		}

		template<typename T>
			requires std::is_integral_v<T>&& std::is_signed_v<T>
		Encoder& push_packed_array(const std::vector<size_t>& dimension_array, const std::span<T> data) {
			int num_type = minimal_signed_bits(std::numeric_limits<T>::max());
			return push_array(dimension_array, data, WXF_HEAD::array, num_type);
		}

		Encoder& push_packed_array(const std::vector<size_t>& dimension_array, const std::span<float> data) {
			return push_array(dimension_array, data, WXF_HEAD::array, 34);
		}

		Encoder& push_packed_array(const std::vector<size_t>& dimension_array, const std::span<double> data) {
			return push_array(dimension_array, data, WXF_HEAD::array, 35);
		}

		Encoder& push_packed_array(const std::vector<size_t>& dimension_array, const std::span<complex_float_t> data) {
			return push_array(dimension_array, data, WXF_HEAD::array, 51);
		}

		Encoder& push_packed_array(const std::vector<size_t>& dimension_array, const std::span<complex_double_t> data) {
			return push_array(dimension_array, data, WXF_HEAD::array, 52);
		}

		template<typename T>
			requires std::is_integral_v<T>
		Encoder& push_numeric_array(const std::vector<size_t>& dimension_array, const std::span<T> data) {
			bool is_signed = std::is_signed_v<T>;
			int num_type;

			if (is_signed)
				num_type = minimal_signed_bits(std::numeric_limits<T>::max());
			else
				num_type = 16 + minimal_unsigned_bits(std::numeric_limits<T>::max());
			return push_array(dimension_array, data, WXF_HEAD::narray, num_type);
		}

		Encoder& push_numeric_array(const std::vector<size_t>& dimension_array, const std::span<float> data) {
			return push_array(dimension_array, data, WXF_HEAD::narray, 34);
		}

		Encoder& push_numeric_array(const std::vector<size_t>& dimension_array, const std::span<double> data) {
			return push_array(dimension_array, data, WXF_HEAD::narray, 35);
		}

		Encoder& push_numeric_array(const std::vector<size_t>& dimension_array, const std::span<complex_float_t> data) {
			return push_array(dimension_array, data, WXF_HEAD::narray, 51);
		}

		Encoder& push_numeric_array(const std::vector<size_t>& dimension_array, const std::span<complex_double_t> data) {
			return push_array(dimension_array, data, WXF_HEAD::narray, 52);
		}
	};

	struct TOKEN {
		WXF_HEAD type;
		int rank = 0;
		union {
			// for number, string, symbol, bigint
			uint64_t length;
			// for array and narray, dimensions[0] is the type, dimensions[1] is the total flatten length
			// so the length is dimensions is rank + 2
			uint64_t* dimensions;
		};
		union { // data
			int64_t i;
			double d;
			int64_t* i_arr;
			uint64_t* u_arr; // only for narray
			double* d_arr;
			complex_float_t* cf_arr;
			complex_double_t* cd_arr;
			char* str;
		};

		TOKEN() : type(WXF_HEAD::i8), rank(0), length(0), i(0) {}

		uint64_t dim(int i) const {
			if (rank > 0)
				return dimensions[i + 2];
			return length;
		}

		void clear() {
			if (type == WXF_HEAD::symbol
				|| type == WXF_HEAD::bigint
				|| type == WXF_HEAD::bigreal
				|| type == WXF_HEAD::string
				|| type == WXF_HEAD::binary_string) {
				free(str);
			}
			else if (type == WXF_HEAD::array || type == WXF_HEAD::narray) {
				free(i_arr);
				free(dimensions);
			}
			// no need to clear i, length, rank, type, as they are just basic types
		}

		~TOKEN() { clear(); }

		// disable copy constructor and copy assignment operator
		TOKEN(const TOKEN&) = delete;
		TOKEN& operator=(const TOKEN&) = delete;

		// move constructor
		TOKEN(TOKEN&& other) noexcept : type(other.type), rank(other.rank), length(other.length), i(other.i) {
			if (type == WXF_HEAD::symbol
				|| type == WXF_HEAD::bigint
				|| type == WXF_HEAD::bigreal
				|| type == WXF_HEAD::string
				|| type == WXF_HEAD::binary_string) {
				str = other.str;
				other.str = nullptr;
			}
			else if (type == WXF_HEAD::array || type == WXF_HEAD::narray) {
				dimensions = other.dimensions;
				i_arr = other.i_arr; // for array, since it is union, we can use i_arr for narray
				other.dimensions = nullptr;
				other.i_arr = nullptr;
			}
		}

		// symbol/bigint/string/binary_string, length, str
		TOKEN(WXF_HEAD t, const std::string_view s) : type(t), rank(0) {
			length = s.size();
			str = (char*)malloc(length + 1);
			std::memcpy(str, s.data(), length);
			str[length] = '\0'; // null-terminate the string
		}

		TOKEN(WXF_HEAD t, const char* s, const size_t len) : type(t), rank(0) {
			length = len;
			str = (char*)malloc(length + 1);
			std::memcpy(str, s, length);
			str[length] = '\0'; // null-terminate the string
		}

		// machine number, val (length is given by the sizeof(val))
		TOKEN(WXF_HEAD t, int8_t v) : type(t), rank(0), length(1), i(v) {}
		TOKEN(WXF_HEAD t, int16_t v) : type(t), rank(0), length(2), i(v) {}
		TOKEN(WXF_HEAD t, int32_t v) : type(t), rank(0), length(3), i(v) {}
		TOKEN(WXF_HEAD t, int64_t v) : type(t), rank(0), length(4), i(v) {}
		TOKEN(WXF_HEAD t, float v) : type(t), rank(0), length(2), d(v) {}
		TOKEN(WXF_HEAD t, double v) : type(t), rank(0), length(4), d(v) {}

		// function, association, delay_rule, rule
		TOKEN(WXF_HEAD t, uint64_t len) : type(t), rank(0), length(len), i(0) {}

		// array/narray
		// with_arr: whether to allocate the i_arr/u_arr, default true
		//	         otherwise, the user may want to use the existing data pointer, 
		//           or only use it for dimensions
		TOKEN(WXF_HEAD t, const std::vector<size_t>& dims, int num_type, size_t len, bool with_arr = true) : type(t) {
			int r = dims.size();
			rank = r;
			dimensions = (uint64_t*)malloc((r + 2) * sizeof(uint64_t));
			dimensions[0] = num_type;
			dimensions[1] = len;
			for (auto i = 0; i < r; i++) {
				dimensions[i + 2] = dims[i];
			}
			if (with_arr) {
				if (num_type == 52) // 128 bits
					cd_arr = (complex_double_t*)malloc(len * sizeof(complex_double_t));
				else // 64 bits
					i_arr = (int64_t*)malloc(len * sizeof(int64_t));
			}
			else
				i_arr = nullptr;
		}

		void to_ustr(Encoder& enc) const {
			const auto& token = *this;

			auto& res = enc.buffer;

			switch (token.type) {
			case WXF_HEAD::i8:
			case WXF_HEAD::i16:
			case WXF_HEAD::i32:
			case WXF_HEAD::i64:
				enc.push_integer(token.i);
				break;
			case WXF_HEAD::f64:
				enc.push_real(token.d);
				break;
			case WXF_HEAD::func:
			case WXF_HEAD::association: {
				res.push_back((uint8_t)token.type);
				serialize_varint(res, token.length);
				break;
			}
			case WXF_HEAD::rule:
			case WXF_HEAD::delay_rule:
				res.push_back((uint8_t)token.type);
				break;
			case WXF_HEAD::symbol:
			case WXF_HEAD::bigint:
			case WXF_HEAD::bigreal:
			case WXF_HEAD::string:
			case WXF_HEAD::binary_string:
				enc.push_string(std::string_view(token.str, token.length), token.type);
				break;
			case WXF_HEAD::array:
			case WXF_HEAD::narray:
				res.push_back((uint8_t)token.type);
				res.push_back(token.dimensions[0]);
				serialize_varint(res, token.rank);
				for (auto i = 0; i < token.rank; i++) {
					serialize_varint(res, token.dimensions[i + 2]);
				}
				if (token.i_arr == nullptr)
					break;
				if (token.dimensions[0] == 3) {
					serialize_binary(res, token.i_arr, token.dimensions[1]);
				}
				else if (token.type == WXF_HEAD::narray && token.dimensions[0] == 19) {
					serialize_binary(res, token.u_arr, token.dimensions[1]);
				}
				else if (token.dimensions[0] == 35) {
					serialize_binary(res, token.d_arr, token.dimensions[1]);
				}
				else if (token.dimensions[0] == 51) {
					serialize_binary(res, token.cf_arr, token.dimensions[1]);
				}
				else if (token.dimensions[0] == 52) {
					serialize_binary(res, token.cd_arr, token.dimensions[1]);
				}
				else {
					std::cerr << "Unsupported number type in packed array or numeric array. " << std::endl;
				}
				break;
			default:
				break;
			}
		}

		std::vector<uint8_t> to_ustr() const {
			Encoder enc;
			to_ustr(enc);
			return enc.buffer;
		}

		// debug only, print the token info
		template<typename T>
		void print(T& ss) const {
			auto& token = *this;
			switch (token.type) {
			case WXF_HEAD::i8:
				ss << "i8: " << token.i << std::endl;
				break;
			case WXF_HEAD::i16:
				ss << "i16: " << token.i << std::endl;
				break;
			case WXF_HEAD::i32:
				ss << "i32: " << token.i << std::endl;
				break;
			case WXF_HEAD::i64:
				ss << "i64: " << token.i << std::endl;
				break;
			case WXF_HEAD::f64:
				ss << "f64: " << token.d << std::endl;
				break;
			case WXF_HEAD::symbol:
				ss << "symbol: " << token.str << std::endl;
				break;
			case WXF_HEAD::bigint:
				ss << "bigint: " << token.str << std::endl;
				break;
			case WXF_HEAD::bigreal:
				ss << "bigreal: " << token.str << std::endl;
				break;
			case WXF_HEAD::string:
				ss << "string: " << token.str << std::endl;
				break;
			case WXF_HEAD::binary_string:
				ss << "binary_string: " << token.str << std::endl;
				break;
			case WXF_HEAD::func:
				ss << "func: " << token.length << " vars" << std::endl;
				break;
			case WXF_HEAD::association:
				ss << "association: " << token.length << " rules" << std::endl;
				break;
			case WXF_HEAD::delay_rule:
				ss << "delay_rule: " << token.length << std::endl;
				break;
			case WXF_HEAD::rule:
				ss << "rule: " << token.length << std::endl;
				break;
			case WXF_HEAD::array: {
				ss << "array: rank = " << token.rank << ", dimensions = ";
				size_t all_len = token.dimensions[1];
				for (int i = 0; i < token.rank; i++) {
					ss << token.dimensions[i + 2] << " ";
				}
				ss << std::endl;

				auto num_type = token.dimensions[0];
				ss << "data: ";
				if (token.i_arr == nullptr)
					break;
				if (num_type < 4) {
					for (size_t i = 0; i < all_len; i++) {
						ss << token.i_arr[i] << " ";
					}
				}
				else if (num_type >= 34 && num_type <= 35) {
					for (size_t i = 0; i < all_len; i++) {
						ss << token.d_arr[i] << " ";
					}
				}
				else if (num_type == 51) {
					for (size_t i = 0; i < all_len; i++) {
						ss << token.cf_arr[i] << " ";
					}
				}
				else if (num_type == 52) {
					for (size_t i = 0; i < all_len; i++) {
						ss << token.cd_arr[i] << " ";
					}
				}
				else {
					std::cerr << "Unknown type:" << num_type << std::endl;
				}
				ss << std::endl;
				break;
			}
			case WXF_HEAD::narray: {
				ss << "narray: rank = " << token.rank << ", dimensions = ";
				for (int i = 0; i < token.rank; i++) {
					ss << token.dimensions[i + 2] << " ";
				}
				ss << std::endl;

				size_t num_type = token.dimensions[0];
				size_t all_len = token.dimensions[1];

				ss << "data: ";
				if (token.i_arr == nullptr)
					break;
				if (num_type >= 16 && num_type < 20) {
					for (size_t i = 0; i < all_len; i++)
						ss << token.u_arr[i] << " ";
				}
				else if (num_type < 4) {
					for (size_t i = 0; i < all_len; i++)
						ss << token.i_arr[i] << " ";
				}
				else if (num_type >= 34 && num_type <= 35) {
					for (size_t i = 0; i < all_len; i++)
						ss << token.d_arr[i] << " ";
				}
				else if (num_type == 51) {
					for (size_t i = 0; i < all_len; i++)
						ss << token.cf_arr[i] << " ";
				}
				else if (num_type == 52) {
					for (size_t i = 0; i < all_len; i++)
						ss << token.cd_arr[i] << " ";
				}
				else {
					std::cerr << "Unknown type: " << num_type << std::endl;
				}
				ss << std::endl;
				break;
			}
			default:
				std::cerr << "Unknown type: " << (int)token.type << std::endl;
			}
		}

		void print() const {
			print(std::cout);
		}
	};

	struct TOKEN_VIEW {
		WXF_HEAD type;
		int rank = 0;
		union {
			// for number, string, symbol, bigint
			size_t length;
			// for array and narray, dimensions[0] is the type, dimensions[1] is the total flatten length
			// so the length is dimensions is rank + 2
			size_t* dimensions;
		};
		const uint8_t* data; // pointer to the data in the original buffer

		TOKEN_VIEW() : type(WXF_HEAD::i8), rank(0), length(0), data(nullptr) {}
		TOKEN_VIEW(const WXF_HEAD t, const size_t len, const uint8_t* d) : type(t), rank(0), length(len), data(d) {}
		TOKEN_VIEW(const WXF_HEAD t, const std::vector<size_t>& dims, const int num_type, const size_t len, const uint8_t* d) : type(t), data(d) {
			int r = dims.size();
			rank = r;
			dimensions = (size_t*)malloc((r + 2) * sizeof(size_t));
			dimensions[0] = num_type;
			dimensions[1] = len;
			for (auto i = 0; i < r; i++) {
				dimensions[i + 2] = dims[i];
			}
		}

		size_t dim(size_t i) const {
			if (rank > 0)
				return dimensions[i + 2];
			return length;
		}

		~TOKEN_VIEW() {
			if (type == WXF_HEAD::array || type == WXF_HEAD::narray)
				free(dimensions);
		}

		TOKEN_VIEW(const TOKEN_VIEW& other) : type(other.type), rank(other.rank), length(other.length), data(other.data) {
			if (type == WXF_HEAD::array || type == WXF_HEAD::narray) {
				dimensions = (size_t*)malloc((rank + 2) * sizeof(size_t));
				std::memcpy(dimensions, other.dimensions, (rank + 2) * sizeof(size_t));
			}
		}

		TOKEN_VIEW(TOKEN_VIEW&& other) noexcept : type(other.type), rank(other.rank), length(other.length), data(other.data) {
			if (type == WXF_HEAD::array || type == WXF_HEAD::narray) {
				dimensions = other.dimensions;
				other.dimensions = nullptr;
			}
		}

		TOKEN_VIEW& operator=(const TOKEN_VIEW& other) {
			if (this != &other) {
				type = other.type;
				rank = other.rank;
				length = other.length;
				data = other.data;
				if (type == WXF_HEAD::array || type == WXF_HEAD::narray) {
					dimensions = (size_t*)malloc((rank + 2) * sizeof(size_t));
					std::memcpy(dimensions, other.dimensions, (rank + 2) * sizeof(size_t));
				}
			}
			return *this;
		}

		TOKEN_VIEW& operator=(TOKEN_VIEW&& other) noexcept {
			if (this != &other) {
				type = other.type;
				rank = other.rank;
				length = other.length;
				data = other.data;
				if (type == WXF_HEAD::array || type == WXF_HEAD::narray) {
					dimensions = other.dimensions;
					other.dimensions = nullptr;
				}
			}
			return *this;
		}

		int64_t get_integer() const {
			if (type == WXF_HEAD::i8)
				return *(int8_t*)data;
			else if (type == WXF_HEAD::i16)
				return *(int16_t*)data;
			else if (type == WXF_HEAD::i32)
				return *(int32_t*)data;
			else if (type == WXF_HEAD::i64)
				return *(int64_t*)data;
			else
				return 0;
		}

		std::string_view get_string_view() const {
			if (type == WXF_HEAD::symbol
				|| type == WXF_HEAD::bigint
				|| type == WXF_HEAD::bigreal
				|| type == WXF_HEAD::string
				|| type == WXF_HEAD::binary_string) {
				return std::string_view((const char*)data, length);
			}
			else
				return std::string_view();
		}

		template<typename T>
		std::span<T> get_arr_span() const {
			if (type != WXF_HEAD::array && type != WXF_HEAD::narray)
				return std::span<T>();
			return std::span<T>((T*)data, dimensions[1]);
		}

		TOKEN to_token(const bool with_arr = true) const {
			TOKEN token;
			token.type = type;
			token.rank = rank;
			if (type == WXF_HEAD::array || type == WXF_HEAD::narray) {
				token.dimensions = (uint64_t*)malloc((rank + 2) * sizeof(uint64_t));
				for (auto i = 0; i < rank + 2; i++) {
					token.dimensions[i] = dimensions[i];
				}
			}
			else
				token.length = length;

			if (type == WXF_HEAD::symbol
				|| type == WXF_HEAD::bigint
				|| type == WXF_HEAD::bigreal
				|| type == WXF_HEAD::string
				|| type == WXF_HEAD::binary_string) {
				token.length = length;
				token.str = (char*)malloc(length + 1);
				std::memcpy(token.str, data, length);
				token.str[length] = '\0';
			}
			else if (type == WXF_HEAD::i8)
				token.i = *(int8_t*)data;
			else if (type == WXF_HEAD::i16)
				token.i = *(int16_t*)data;
			else if (type == WXF_HEAD::i32)
				token.i = *(int32_t*)data;
			else if (type == WXF_HEAD::i64)
				token.i = *(int64_t*)data;
			else if (type == WXF_HEAD::f64)
				token.d = *(double*)data;
			else if (type == WXF_HEAD::array || type == WXF_HEAD::narray) {
				if (with_arr) {
					auto num_type = dimensions[0];
					auto all_len = dimensions[1];

					if (num_type == 52) // 128 bits
						token.cd_arr = (complex_double_t*)malloc(all_len * sizeof(complex_double_t));
					else // 64 bits
						token.i_arr = (int64_t*)malloc(all_len * sizeof(int64_t));

#define LOAD_ARRAY_TMP(TYPE1, TYPE2, FIELD) do { \
						const uint8_t* buffer = data; \
						for (size_t i = 0; i < all_len; i++) { \
							token.FIELD[i] = static_cast<TYPE2>(*(TYPE1*)buffer); \
							buffer += sizeof(TYPE1); \
						} \
					} while (0)

					switch (num_type) {
					case 0: LOAD_ARRAY_TMP(int8_t, int64_t, i_arr); break;
					case 1: LOAD_ARRAY_TMP(int16_t, int64_t, i_arr); break;
					case 2: LOAD_ARRAY_TMP(int32_t, int64_t, i_arr); break;
					case 3: LOAD_ARRAY_TMP(int64_t, int64_t, i_arr); break;
					case 16: LOAD_ARRAY_TMP(uint8_t, uint64_t, u_arr); break;
					case 17: LOAD_ARRAY_TMP(uint16_t, uint64_t, u_arr); break;
					case 18: LOAD_ARRAY_TMP(uint32_t, uint64_t, u_arr); break;
					case 19: LOAD_ARRAY_TMP(uint64_t, uint64_t, u_arr); break;
					case 34: LOAD_ARRAY_TMP(float, double, d_arr); break;
					case 35: LOAD_ARRAY_TMP(double, double, d_arr); break;
					case 51: LOAD_ARRAY_TMP(complex_float_t, complex_float_t, cf_arr); break;
					case 52: LOAD_ARRAY_TMP(complex_double_t, complex_double_t, cd_arr); break;
					default:
						std::cerr << "Unsupported number type in packed array or numeric array. " << std::endl;
						break;
					}
#undef LOAD_ARRAY_TMP

				}
			}
			return token;
		}

	};

	struct Parser {
		const uint8_t* buffer; // the buffer to read
		size_t pos = 0;
		size_t size = 0; // the size of the buffer
		int err = 0; // 0 is ok, otherwise error
		int view_mode = 0; // 0 is view mode, 1 is normal mode
		std::vector<TOKEN_VIEW> token_views;
		std::vector<TOKEN> tokens;

		Parser(const uint8_t* buf, const size_t len, int view_mode = 0) : buffer(buf), pos(0), size(len), err(0), view_mode(view_mode) {}
		Parser(const std::vector<uint8_t>& buf, int view_mode = 0) : buffer(buf.data()), pos(0), size(buf.size()), err(0), view_mode(view_mode) {}
		Parser(const std::string_view buf, int view_mode = 0) : buffer((const uint8_t*)buf.data()), pos(0), size(buf.size()), err(0), view_mode(view_mode) {}

		// default special member functions
		Parser() = default;
		~Parser() = default;
		Parser(const Parser&) = default;
		Parser& operator=(const Parser&) = default;
		Parser(Parser&&) noexcept = default;
		Parser& operator=(Parser&&) noexcept = default;

		inline uint64_t read_varint() {
			const uint8_t* ptr = buffer + pos;
			const uint8_t* end = buffer + size;
			uint64_t result = 0;
			uint8_t b;

			if (ptr >= end) return 0;

			b = *ptr++; result = uint64_t(b & 0x7F);         if (!(b & 0x80) || ptr >= end) goto done;
			b = *ptr++; result |= uint64_t(b & 0x7F) << 7;   if (!(b & 0x80) || ptr >= end) goto done;
			b = *ptr++; result |= uint64_t(b & 0x7F) << 14;  if (!(b & 0x80) || ptr >= end) goto done;
			b = *ptr++; result |= uint64_t(b & 0x7F) << 21;  if (!(b & 0x80) || ptr >= end) goto done;
			b = *ptr++; result |= uint64_t(b & 0x7F) << 28;  if (!(b & 0x80) || ptr >= end) goto done;
			b = *ptr++; result |= uint64_t(b & 0x7F) << 35;  if (!(b & 0x80) || ptr >= end) goto done;
			b = *ptr++; result |= uint64_t(b & 0x7F) << 42;  if (!(b & 0x80) || ptr >= end) goto done;
			b = *ptr++; result |= uint64_t(b & 0x7F) << 49;  if (!(b & 0x80) || ptr >= end) goto done;
			b = *ptr++; result |= uint64_t(b & 0x7F) << 56;  if (!(b & 0x80) || ptr >= end) goto done;
			b = *ptr++; result |= uint64_t(b & 0x7F) << 63;
		done:
			pos = ptr - buffer;
			return result;
		}

		void parse() {
			// check the file head
			if (pos == 0) {
				if (size < 2 || buffer[0] != 56 || buffer[1] != 58) {
					std::cerr << "Invalid WXF file" << std::endl;
					err = 1;
					return;
				}
				pos = 2;
			}

			while (pos < size) {
				WXF_HEAD type = (WXF_HEAD)(buffer[pos]); pos++;

				if (pos == size)
					break;

				switch (type) {
				case WXF_HEAD::i8:
				case WXF_HEAD::i16:
				case WXF_HEAD::i32:
				case WXF_HEAD::i64:
				case WXF_HEAD::f64: {
					auto length = size_of_head_num_type(type);
					token_views.emplace_back(type, length, buffer + pos);
					pos += length;
					break;
				}
				case WXF_HEAD::symbol:
				case WXF_HEAD::bigint:
				case WXF_HEAD::bigreal:
				case WXF_HEAD::string:
				case WXF_HEAD::binary_string: {
					auto length = read_varint();
					token_views.emplace_back(type, length, buffer + pos);
					pos += length;
					break;
				}
				case WXF_HEAD::func:
				case WXF_HEAD::association: {
					auto length = read_varint();
					token_views.emplace_back(type, length, buffer + pos);
					break;
				}
				case WXF_HEAD::delay_rule:
				case WXF_HEAD::rule:
					token_views.emplace_back(type, size_t(2), buffer + pos);
					break;
				case WXF_HEAD::array:
				case WXF_HEAD::narray: {
					int num_type = read_varint();
					auto r = read_varint();
					std::vector<size_t> dims(r);
					size_t all_len = 1;
					for (size_t i = 0; i < r; i++) {
						dims[i] = read_varint();
						all_len *= dims[i];
					}
					token_views.emplace_back(type, dims, num_type, all_len, buffer + pos);
					pos += all_len * size_of_arr_num_type(num_type);
					break;
				}
				default:
					std::cerr << "Unknown head type: " << (int)type << " pos: " << pos << std::endl;
					err = 3;
					break;
				}
			}
			err = 0;

			if (view_mode == 1) {
				tokens.reserve(token_views.size());
				for (auto& t : token_views)
					tokens.emplace_back(t.to_token());
				// clear the token_views to save memory
				token_views = std::vector<TOKEN_VIEW>();
			}
		}
	};

	struct ExprNode {
		size_t index; // the index of the token in the tokens vector
		size_t size; // the size of the children
		std::unique_ptr<ExprNode[]> children; // the children of the node
		WXF_HEAD type;

		ExprNode() : index(0), size(0), children(nullptr), type(WXF_HEAD::i8) {} // default constructor

		ExprNode(size_t idx, size_t sz, WXF_HEAD t) : index(idx), size(sz), type(t) {
			constexpr size_t MAX_ALLOC = std::numeric_limits<int64_t>::max();
			if (size > MAX_ALLOC) {
				throw std::bad_alloc();
			}
			else if (size > 0) {
				children = std::make_unique<ExprNode[]>(size);
			}
		}

		ExprNode(const ExprNode&) = delete; // disable copy constructor
		ExprNode& operator=(const ExprNode&) = delete; // disable copy assignment operator
		// move constructor
		ExprNode(ExprNode&& other) noexcept : index(other.index), size(other.size),
			children(std::move(other.children)), type(other.type) {
			other.index = 0;
			other.size = 0;
			other.children = nullptr;
			other.type = WXF_HEAD::i8;
		}

		// move assignment operator
		ExprNode& operator=(ExprNode&& other) noexcept {
			if (this != &other) {
				index = other.index;
				size = other.size;
				children = std::move(other.children);
				type = other.type;
				other.size = 0;
				other.index = 0;
				other.children = nullptr;
				other.type = WXF_HEAD::i8;
			}
			return *this;
		}

		// destructor
		void clear() {
			index = 0;
			size = 0;
			type = WXF_HEAD::i8;
			if (children) {
				children.reset();
			}
		}

		~ExprNode() { clear(); }
		bool has_children() const { return size > 0; }
		const ExprNode& operator[] (size_t i) const { return children[i]; }
		ExprNode& operator[] (size_t i) { return children[i]; }
	};

	void node_to_ustr(Encoder& enc, const std::vector<TOKEN>& tokens, const ExprNode& node) {
		auto& token = tokens[node.index];
		auto& res = enc.buffer;

		switch (node.type) {
		case WXF_HEAD::func:
		case WXF_HEAD::association:
			res.push_back((uint8_t)node.type);
			serialize_varint(res, node.size);
			res.push_back((uint8_t)token.type);
			serialize_varint(res, token.length);
			res.insert(res.end(), token.str, token.str + token.length);
			for (size_t i = 0; i < node.size; i++) {
				node_to_ustr(enc, tokens, node.children[i]);
			}
			break;
		case WXF_HEAD::rule:
		case WXF_HEAD::delay_rule:
			res.push_back((uint8_t)node.type);
			for (size_t i = 0; i < node.size; i++) {
				node_to_ustr(enc, tokens, node.children[i]);
			}
			break;
		default:
			token.to_ustr(enc);
			break;
		}
	}

	struct ExprTree {
		std::vector<TOKEN> tokens;
		ExprNode root;

		ExprTree() {} // default constructor
		ExprTree(Parser parser, size_t index, size_t size, WXF_HEAD type) : root(index, size, type) {
			tokens = std::move(parser.tokens);
		}

		ExprTree(const ExprTree&) = delete; // disable copy constructor
		ExprTree& operator=(const ExprTree&) = delete; // disable copy assignment operator

		// move constructor
		ExprTree(ExprTree&& other) noexcept : tokens(std::move(other.tokens)), root(std::move(other.root)) {
			other.root.size = 0;
			other.root.index = 0;
			other.root.children = nullptr;
		}

		// move assignment operator
		ExprTree& operator=(ExprTree&& other) noexcept {
			if (this != &other) {
				tokens = std::move(other.tokens);
				root = std::move(other.root);
				other.root.size = 0;
				other.root.index = 0;
				other.root.children = nullptr;
			}
			return *this;
		}

		const TOKEN& operator[](const ExprNode& node) const {
			return tokens[node.index];
		}

		// it is super slow...
		// TODO: optimize this function
		std::vector<uint8_t> to_ustr(bool include_head = true) const {
			Encoder enc;
			if (include_head) {
				enc.buffer.push_back(56); // WXF head
				enc.buffer.push_back(58); // WXF head
			}
			node_to_ustr(enc, tokens, root);
			return enc.buffer;
		}
	};

	ExprTree MakeExprTree(Parser& parser) {
		ExprTree tree;
		if (parser.err != 0)
			return tree;

		// convert token views to tokens if in view mode
		if (parser.view_mode == 0) {
			tree.tokens.reserve(parser.token_views.size());
			for (auto& t : parser.token_views)
				tree.tokens.emplace_back(t.to_token());
		}
		else {
			tree.tokens = std::move(parser.tokens);
		}

		auto total_len = tree.tokens.size();
		auto& tokens = tree.tokens;

		std::vector<ExprNode*> expr_stack; // the stack to store the current father nodes
		std::vector<size_t> node_stack; // the vector to store the node index

		std::function<void(void)> move_to_next_node = [&]() {
			if (node_stack.empty())
				return;

			node_stack.back()++; // move to the next node
			if (node_stack.back() >= expr_stack.back()->size) {
				expr_stack.pop_back(); // pop the current node
				node_stack.pop_back(); // pop the current node index
				move_to_next_node();
			}
			};

		// first we need to find the root node
		size_t pos = 0;
		auto& token = tokens[pos];
		if (token.type == WXF_HEAD::func) {
			// i + 1 is the head of the function (a symbol)
			tree.root = ExprNode(pos + 1, token.length, token.type);
			pos += 2; // skip the head
		}
		else if (token.type == WXF_HEAD::association) {
			// association does not have a head
			tree.root = ExprNode(pos + 1, token.length, token.type);
			pos += 1;
		}
		else {
			// if the token is not a function type, only one token is allowed
			tree.root = ExprNode(pos, 0, token.type);
			return tree;
		}

		expr_stack.push_back(&(tree.root));
		node_stack.push_back(0);

		// now we need to parse the expression
		for (; pos < total_len; pos++) {
			auto& token = tokens[pos];
			if (token.type == WXF_HEAD::func || token.type == WXF_HEAD::association) {
				// if the token is a function type, we need to create a new node
				auto node_pos = node_stack.back();
				auto parent = expr_stack.back();
				auto& node = parent->children[node_pos];
				if (token.type == WXF_HEAD::func) {
					node = ExprNode(pos + 1, token.length, token.type);
					pos++; // skip the head
				}
				else
					node = ExprNode(pos, token.length, token.type);
				expr_stack.push_back(&(node)); // push the new node to the stack
				node_stack.push_back(0); // push the new node index to the stack
			}
			else if (token.type == WXF_HEAD::delay_rule || token.type == WXF_HEAD::rule) {
				// if the token is a rule type, we need to create a new node
				auto node_pos = node_stack.back();
				auto parent = expr_stack.back();
				auto& node = parent->children[node_pos];
				node = ExprNode(pos, 2, token.type);
				expr_stack.push_back(&(node)); // push the new node to the stack
				node_stack.push_back(0); // push the new node index to the stack
			}
			else {
				// if the token is not a function type, we need to move to the next node
				auto node_pos = node_stack.back();
				auto parent = expr_stack.back();
				auto& node = parent->children[node_pos];
				node = ExprNode(pos, 0, token.type);

				move_to_next_node();
			}
		}

		if (!node_stack.empty()) {
			std::cerr << "Error: not all nodes are parsed" << std::endl;
			for (auto& node : expr_stack) {
				node->clear();
			}
		}

		return tree;
	}

	ExprTree MakeExprTree(const uint8_t* str, const size_t len) {
		Parser parser(str, len, 1);
		parser.parse();
		return MakeExprTree(parser);
	}

	ExprTree MakeExprTree(const std::vector<uint8_t>& str) {
		Parser parser(str, 1);
		parser.parse();
		return MakeExprTree(parser);
	}

	ExprTree MakeExprTree(const std::string_view str) {
		Parser parser(reinterpret_cast<const uint8_t*>(str.data()), str.size(), 1);
		parser.parse();
		return MakeExprTree(parser);
	}

	ExprTree MakeExprTree(const std::filesystem::path filename) {
		if (!std::filesystem::exists(filename)) {
			std::cerr << "Error: File does not exist!" << std::endl;
			return ExprTree();
		}
		ExprTree expr_tree;
		std::ifstream file(filename, std::ios::binary | std::ios::ate);
		std::streamsize size = file.tellg();
		file.seekg(0, std::ios::beg);

		std::vector<uint8_t> buffer(size);
		if (!file.read((char*)buffer.data(), size)) {
			std::cerr << "Failed to read file!" << std::endl;
			return ExprTree();
		}

		return MakeExprTree(buffer);
	}

	/***********************************************************************************/

	// a simple FullForm parser, we only support [0-9,a-z,A-Z,$] in symbol names
	// it is used to write a template engine to generate WXF files
	// and we add a special expression type starting with # for subexpression labels
	// e.g. #x for a integer x, e.g. #1, #2, ...
	// and then we can replace these labels with actual expressions when generating WXF files
	// do not use for a very complex FullForm parsing, it is not efficient and robust enough
	namespace FullForm {
		enum class atom_type {
			Integer,
			Real,
			String,
			Symbol,
			Expression,
			Null
		};

		struct atom_expression {
			atom_type type_;
			std::string value_;

			atom_expression(atom_type type, const std::string_view value)
				: type_(type), value_(value) {
			}

			const atom_type get_type() const { return type_; }
			const std::string& get_value() const { return value_; }

			std::string to_FullForm() const {
				switch (type_) {
				case atom_type::String:
					return "\"" + value_ + "\"";
				default:
					return value_;
				}
			}
		};

		struct lexer {
			std::string input_;
			size_t position_;
			size_t length_;

			char current_char() const {
				return position_ < length_ ? input_[position_] : '\0';
			}

			void advance() {
				if (position_ < length_) position_++;
			}

			void skip_ws() {
				while (position_ < length_ && std::isspace(static_cast<unsigned char>(current_char()))) {
					advance();
				}
			}

			enum token_type {
				IDENTIFIER,    // 
				EXPRESSION,    // start with #, use as a label of subexpression, it is not standard mathematica 
				INTEGER,       // 
				REAL,          // 
				STRING,        // 
				LBRACKET,      // [
				RBRACKET,      // ]
				COMMA,         // ,
				END            // 
			};

			struct token {
				token_type type;
				std::string value;
				size_t position;
			};

			lexer(const std::string_view input)
				: input_(input), position_(0), length_(input.length()) {
			}

			token nextToken() {
				skip_ws();

				if (position_ >= length_) {
					return { END, "", position_ };
				}

				size_t startPos = position_;
				char ch = current_char();

				// $ 
				if (std::isalpha(ch) || ch == '$') {
					std::string value;
					while (position_ < length_ &&
						(std::isalnum(static_cast<unsigned char>(current_char())) || current_char() == '$')) {
						value += current_char();
						advance();
					}
					return { IDENTIFIER, value, startPos };
				}

				// expression starting with #
				if (ch == '#') {
					std::string value;
					value += ch;
					advance();
					while (position_ < length_ &&
						(std::isalnum(static_cast<unsigned char>(current_char())) || current_char() == '$')) {
						value += current_char();
						advance();
					}
					return { EXPRESSION, value, startPos };
				}

				// number
				if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '.' || ch == '-') {
					std::string value;
					bool hasDot = false;
					bool hasE = false;

					// minus sign
					if (ch == '-') {
						value += ch;
						advance();
						if (position_ >= length_ || !std::isdigit(static_cast<unsigned char>(current_char()))) {
							// single minus sign, return as IDENTIFIER
							return { IDENTIFIER, value, startPos };
						}
						ch = current_char();
					}

					// integer part
					while (position_ < length_ && std::isdigit(static_cast<unsigned char>(current_char()))) {
						value += current_char();
						advance();
					}

					// decimal part
					if (position_ < length_ && current_char() == '.') {
						hasDot = true;
						value += '.';
						advance();
						while (position_ < length_ && std::isdigit(static_cast<unsigned char>(current_char()))) {
							value += current_char();
							advance();
						}
					}

					// scientific notation
					if (position_ < length_ && (current_char() == 'e' || current_char() == 'E')) {
						hasE = true;
						value += current_char();
						advance();

						if (position_ < length_ && (current_char() == '+' || current_char() == '-')) {
							value += current_char();
							advance();
						}

						if (position_ >= length_ || !std::isdigit(static_cast<unsigned char>(current_char()))) {
							std::cerr << "Invalid scientific notation at position " << position_ << std::endl;
							return { END, "", startPos};
						}

						while (position_ < length_ && std::isdigit(static_cast<unsigned char>(current_char()))) {
							value += current_char();
							advance();
						}
					}

					if (hasDot || hasE) {
						return { REAL, value, startPos };
					}
					else {
						return { INTEGER, value, startPos };
					}
				}

				// string
				if (ch == '"') {
					std::string value;
					advance();

					while (position_ < length_ && current_char() != '"') {
						if (current_char() == '\\' && position_ + 1 < length_) {
							advance();
							char next = current_char();
							switch (next) {
							case 'n': value += '\n'; break;
							case 't': value += '\t'; break;
							case 'r': value += '\r'; break;
							case '"': value += '"'; break;
							case '\\': value += '\\'; break;
							default: value += '\\'; value += next; break;
							}
						}
						else {
							value += current_char();
						}
						advance();
					}

					if (position_ >= length_ || current_char() != '"') {
						std::cerr << "Unterminated string at position " << startPos << std::endl;
						return { END, "", startPos };
					}
					advance();  // skip closing "

					return { STRING, value, startPos };
				}

				// single-character tokens
				switch (ch) {
				case '[':
					advance();
					return { LBRACKET, "[", startPos };
				case ']':
					advance();
					return { RBRACKET, "]", startPos };
				case ',':
					advance();
					return { COMMA, ",", startPos };
				}

				std::cerr << "Unknown character '" << ch << "' at position " << position_ << std::endl;
			}
		};

		struct expression {
			atom_expression head_;
			std::vector<expression> args_;

			expression(const atom_expression& head) : head_(head), args_() {}
			expression(const atom_expression& head, const std::vector<expression>& args) : head_(head), args_(args) {}
			bool is_atom() const { return args_.size() == 0; }

			std::string to_FullForm() const {
				if (is_atom()) {
					return head_.to_FullForm();
				}
				else {
					std::string result = head_.to_FullForm() + "[";
					for (size_t i = 0; i < args_.size(); ++i) {
						result += args_[i].to_FullForm();
						if (i + 1 < args_.size()) {
							result += ", ";
						}
					}
					result += "]";
					return result;
				}
			}
		};

		struct parser {
			lexer lexer_;
			lexer::token currentToken_;

			void consume(lexer::token_type expected) {
				if (currentToken_.type == expected) {
					currentToken_ = lexer_.nextToken();
				}
				else {
					std::cerr << "Parse Error: Unexpected token '" << currentToken_.value
						<< "', expected " << static_cast<int>(expected)
						<< " at position " << currentToken_.position << std::endl;
				}
			}

			atom_expression parse_atom() {
				std::string value = currentToken_.value;

				switch (currentToken_.type) {
				case lexer::IDENTIFIER: {
					consume(lexer::IDENTIFIER);
					return atom_expression(atom_type::Symbol, value);
				}
				case lexer::EXPRESSION: {
					consume(lexer::EXPRESSION);
					return atom_expression(atom_type::Expression, value);
				}
				case lexer::INTEGER: {
					consume(lexer::INTEGER);
					return atom_expression(atom_type::Integer, value);
				}
				case lexer::REAL: {
					consume(lexer::REAL);
					return atom_expression(atom_type::Real, value);
				}
				case lexer::STRING: {
					consume(lexer::STRING);
					return atom_expression(atom_type::String, value);
				}
				default:
					std::cerr << "Parse Error: Unexpected token '" << currentToken_.value
						<< "' at position " << currentToken_.position << std::endl;
					return atom_expression(atom_type::Null, "");
				}
			}

			expression parse_expression() {
				atom_expression head = parse_atom();

				if (currentToken_.type == lexer::LBRACKET) {
					consume(lexer::LBRACKET);
					std::vector<expression> args;

					if (currentToken_.type != lexer::RBRACKET) {
						args.push_back(parse_expression());

						while (currentToken_.type == lexer::COMMA) {
							consume(lexer::COMMA);
							args.push_back(parse_expression());
						}
					}

					// if we meet expr like f[], we use a null atom as argument
					if (args.size() == 0) {
						args.push_back(expression(atom_expression(atom_type::Null, "")));
					}

					consume(lexer::RBRACKET);
					return expression(head, args);
				}

				return head;
			}

			parser(const std::string_view input) : lexer_(input) {
				currentToken_ = lexer_.nextToken();
			}

			expression parse() {
				expression result = parse_expression();

				if (currentToken_.type != lexer::END) {
					std::cerr << "Parse Error: Unexpected token at end: " << currentToken_.value << std::endl;
				}

				return result;
			}
		};

		// we allow use { }, so we need to convert { } to List[ ]
		inline expression parse_FullForm(const std::string_view str) {
			std::string mod_str;
			mod_str.reserve(str.size() + 10);
			for (auto c : str) {
				if (c == '{') 
					mod_str += "List[";
				else if (c == '}') 
					mod_str += "]";
				else 
					mod_str += c;
			}

			parser parser(mod_str);
			return parser.parse();
		}
	}

	// we allow use a map to store function that generating sub-expressions
	void fullform_to_wxf(Encoder& encoder, const FullForm::expression& expr, 
		const std::unordered_map<std::string, std::function<void(Encoder&)>>& map) {

		if (expr.is_atom()) {
				switch (expr.head_.get_type()) {
				case FullForm::atom_type::Integer:
					encoder.push_integer(std::stoll(expr.head_.get_value()));
					break;
				case FullForm::atom_type::Real:
					encoder.push_real(std::stod(expr.head_.get_value()));
					break;
				case FullForm::atom_type::String:
					encoder.push_string(expr.head_.get_value());
					break;
				case FullForm::atom_type::Symbol:
					encoder.push_symbol(expr.head_.get_value());
					break;
				case FullForm::atom_type::Null:
					break;
				case FullForm::atom_type::Expression: {
					auto& vv = expr.head_.get_value();
					auto it = map.find(vv);
					if (it != map.end())
						it->second(encoder);
					else
						std::cerr << "Error: expression id " << vv << " not found in map." << std::endl;
					break;
				}
				default:
					std::cerr << "Error: unknown atom type." << std::endl;
					break;
				}
			}
		else {
			size_t len = expr.args_.size();
			if (expr.args_[0].is_atom() &&
				expr.args_[0].head_.get_type() == FullForm::atom_type::Null) {
				len = 0;
			}
			std::string_view name = expr.head_.get_value();

			// rule is special, its length is always 2 and omited by WXF encoder
			if (name == "Rule")
				encoder.push_rule();
			else if (name == "RuleDelayed")
				encoder.push_delay_rule();
			else
				encoder.push_function(name, len);

			for (size_t i = 0; i < len; i++) {
				fullform_to_wxf(encoder, expr.args_[i], map);
			}
		}
	}

	void fullform_to_wxf(Encoder& encoder, const FullForm::expression& expr,
		const std::unordered_map<std::string, Encoder>& map) {
		std::unordered_map<std::string, std::function<void(Encoder&)>> func_map;
		for (const auto& [key, value] : map) {
			func_map[key] = [value](Encoder& enc) {
				enc.push_ustr(value.buffer);
			};
		}
		fullform_to_wxf(encoder, expr, func_map);
	}

	template<typename MapType>
	Encoder fullform_to_wxf(const std::string_view ff_template, const MapType& map, bool include_head = true) {
		Encoder encoder;
		encoder.buffer.reserve(ff_template.size() * 32); // reserve some space
		if (include_head) {
			encoder.buffer.push_back(56); // WXF head
			encoder.buffer.push_back(58); // WXF head
		}
		fullform_to_wxf(encoder, FullForm::parse_FullForm(ff_template), map);
		return encoder;
	}

} // namespace WXF_PARSER