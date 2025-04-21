/*
	Copyright (C) 2025 Zhenjie Li (Li, Zhenjie)

	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
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

	We here intend to support import and export a SparseArray expression with rational
	entries, so some types are not supported, such as complex numbers.
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
	-		193			packed array
	-		194			numeric array
	*		65			association
	*		58			delayed rule in association
	*		45			rule in association

	* is supported, - is partially supported
*/

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <cstring>
#include <variant>

namespace WXF_PARSER {

	enum WXF_HEAD {
		// function type
		func = 102,
		association = 65,
		// char type
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

	using NumberType = std::variant<int8_t, int16_t, int32_t, int64_t,
								uint8_t, uint16_t, uint32_t, uint64_t,
								float, double, bool>;

	NumberType select_type(int index) {
		switch (index) {
			case 0: return int8_t(0);
			case 1: return int16_t(0);
			case 2: return int32_t(0);
			case 3: return int64_t(0);
			case 16: return uint8_t(0);
			case 17: return uint16_t(0);
			case 18: return uint32_t(0);
			case 19: return uint64_t(0);
			case 34: return float(0);
			case 35: return double(0);
			default:
				std::cerr << "Unsupported type index" << std::endl;
				return bool(0);
		}
	}

	struct WXF_TOKEN {
		WXF_HEAD type;
		int rank;
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
			int64_t* i_arr; // for array
			uint64_t* u_arr; // only for narray
			double* d_arr; // for array and narray, but not fully supported yet
			char* str; 
		};

		WXF_TOKEN() : type(WXF_HEAD::i8), rank(0), length(0), i(0) {}

		void clear() {
			if (type == WXF_HEAD::symbol 
				|| type == WXF_HEAD::bigint
				|| type == WXF_HEAD::bigreal
				|| type == WXF_HEAD::string
				|| type == WXF_HEAD::binary_string) {
				if (str == nullptr)
					return;

				delete[] str;
			}
			else if (type == WXF_HEAD::array) {
				if (dimensions == nullptr)
					return;

				delete[] dimensions;
				delete[] i_arr;
			}
			else if (type == WXF_HEAD::narray) {
				if (dimensions == nullptr)
					return;

				if (dimensions[0] >= 16 && dimensions[0] <= 20) { // unsigned type
					delete[] u_arr;
				}
				else if (dimensions[0] >= 0 && dimensions[0] <= 3) { // signed type
					delete[] i_arr;
				}
				delete[] dimensions;
			}
			// no need to clear i, length, rank, type, as they are just basic types
		}

		~WXF_TOKEN() { clear(); }

		// disable copy constructor and copy assignment operator
		WXF_TOKEN(const WXF_TOKEN&) = delete; 
		WXF_TOKEN& operator=(const WXF_TOKEN&) = delete; 

		// move constructor
		WXF_TOKEN(WXF_TOKEN&& other) noexcept : type(other.type), rank(other.rank), length(other.length), i(other.i) {
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

	};


	struct Parser {
		uint8_t* buffer = nullptr; // the buffer to read
		size_t pos = 0;
		size_t size = 0; // the size of the buffer
		std::vector<WXF_TOKEN> tokens; // the tokens read from the buffer

		Parser(uint8_t* buf, const size_t len) : buffer(buf), pos(0), size(len) {}
		Parser(std::vector<uint8_t>& buf) : buffer(buf.data()), pos(0), size(buf.size()) {}

		// we suppose that the length does not exceed 2^64 - 1 .. 
		uint64_t ReadVarint() {
			size_t count = 0;
			uint64_t result = 0;
			auto ptr = buffer + pos;

			while (pos < size && count < 8) {
				result |= (uint64_t)((*ptr) & 127) << (7 * count);
				count++; pos++;
				if (!((*ptr) & 128))
					break;
				ptr++;
			}

			return result;
		}

		void parseExpr() {
			// check the file head
			if (pos == 0) {
				if (size < 2 || buffer[0] != 56 || buffer[1] != 58) {
					std::cerr << "Invalid WXF file" << std::endl;
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
						tokens.push_back(makeNumber<int8_t>());
						break;
					case WXF_HEAD::i16:
						tokens.push_back(makeNumber<int16_t>());
						break;
					case WXF_HEAD::i32:
						tokens.push_back(makeNumber<int32_t>());
						break;
					case WXF_HEAD::i64:
						tokens.push_back(makeNumber<int64_t>());
						break;
					case WXF_HEAD::f64:
						tokens.push_back(makeNumber<double>());
						break;
					case WXF_HEAD::symbol: 
						tokens.push_back(makeString(type));
						break;
					case WXF_HEAD::bigint:
						tokens.push_back(makeString(type));
						break;
					case WXF_HEAD::bigreal:
						tokens.push_back(makeString(type));
						break;
					case WXF_HEAD::string:
						tokens.push_back(makeString(type));
						break;
					case WXF_HEAD::binary_string:
						tokens.push_back(makeString(type));
						break;
					case WXF_HEAD::func: 
						tokens.push_back(makeFunction(type));
						break;
					case WXF_HEAD::association:
						tokens.push_back(makeFunction(type));
						break;
					case WXF_HEAD::delay_rule:
						tokens.push_back(makeRule(type));
						break;
					case WXF_HEAD::rule:
						tokens.push_back(makeRule(type));
						break;
					case WXF_HEAD::array: 
						tokens.push_back(makeArray());
						break;
					case WXF_HEAD::narray:
						tokens.push_back(makeNArray());
						break;
					default:
						std::cerr << "Unknown supported head type: " << (int)type << " pos: " << pos << std::endl;
						break;
				}
			}
		}

		// machine number, val (length is given by the sizeof(val))
		template <typename T>
		WXF_TOKEN makeNumber() {
			WXF_TOKEN node;
			node.length = 0;

			T val;
			std::memcpy(&val, buffer + pos, sizeof(val));
			
			if constexpr (std::is_same_v<T, double>) {
				node.d = val;
			}
			else {
				node.i = (int64_t)val;
			}
			
			pos += sizeof(T) / sizeof(uint8_t);

			if constexpr (std::is_same_v<T, int8_t>) {
				node.type = WXF_HEAD::i8;
			}
			else if constexpr (std::is_same_v<T, int16_t>) {
				node.type = WXF_HEAD::i16;
			}
			else if constexpr (std::is_same_v<T, int32_t>) {
				node.type = WXF_HEAD::i32;
			}
			else if constexpr (std::is_same_v<T, int64_t>) {
				node.type = WXF_HEAD::i64;
			}
			else if constexpr (std::is_same_v<T, double>) {
				// it is better to use float64_t (in C++23)
				node.type = WXF_HEAD::f64;
			}

			return node;
		}

		// symbol/bigint/string/binary_string, length, str
		WXF_TOKEN makeString(WXF_HEAD type) {
			WXF_TOKEN node;
			node.type = type;
			node.length = ReadVarint();
			node.str = new char[node.length + 1];
			std::memcpy(node.str, buffer + pos, node.length);
			node.str[node.length] = '\0'; // add null terminator
			pos += node.length;
			return node;
		}

		// func/association, length
		WXF_TOKEN makeFunction(WXF_HEAD type) {
			WXF_TOKEN node;
			node.type = type;
			node.length = ReadVarint();
			return node;
		}

		WXF_TOKEN makeRule(WXF_HEAD type) {
			WXF_TOKEN node;
			node.type = type;
			node.length = 2;
			return node;
		}

		// array, rank, dimensions, data
		// for the num_type
		// 0 is int8_t      1 is int16_t
		// 2 is int32_t     3 is int64_t
		// 34 float         35 double
		// 51 complex float 52 complex double
		// we only support int8_t, int16_t, int32_t, int64_t, float, double

		WXF_TOKEN makeArray() {
			auto num_type = (int)ReadVarint();
			if (num_type > 50) {
				std::cerr << "Unsupported type: " << num_type << std::endl;
				return WXF_TOKEN();
			}
			size_t size_of_type;
			if (num_type >= 34 && num_type <= 52) {
				size_of_type = (size_t)1 << (num_type - 32);
			}
			else {
				size_of_type = (size_t)1 << num_type;
			}

			WXF_TOKEN node;
			node.type = WXF_HEAD::array;
			node.rank = (int)ReadVarint();
			node.dimensions = new uint64_t[node.rank + 2];
			size_t all_len = 1;
			for (auto i = 0; i < node.rank; i++) {
				node.dimensions[i + 2] = ReadVarint();
				all_len *= node.dimensions[i + 2];
			}
			node.dimensions[0] = num_type;
			node.dimensions[1] = all_len;
			if (num_type >= 34 && num_type <= 52) {
				node.d_arr = new double[all_len];
				std::visit([&](auto&& x) {
					using T = std::decay_t<decltype(x)>;
					T val;
					for (size_t i = 0; i < all_len; i++) {
						std::memcpy(&val, buffer + pos, size_of_type * sizeof(uint8_t));
						node.d_arr[i] = (double)val;
						pos += size_of_type;
					}
					}
				, select_type(num_type));
			}
			else {
				node.i_arr = new int64_t[all_len];
				std::visit([&](auto&& x) {
					using T = std::decay_t<decltype(x)>;
					T val;
					for (size_t i = 0; i < all_len; i++) {
						std::memcpy(&val, buffer + pos, size_of_type * sizeof(uint8_t));
						node.i_arr[i] = (int64_t)val;
						pos += size_of_type;
					}
					}
				, select_type(num_type));
			}

			return node;
		}

		// narray, rank, dimensions, data
		// for the num_type
		// 0 is int8_t      1 is int16_t
		// 2 is int32_t     3 is int64_t
		// 16 is uint8_t    17 is uint16_t
		// 18 is uint32_t   19 is uint64_t
		// 34 float         35 double
		// 51 complex float 52 complex double
		// we only support int8_t, int16_t, int32_t, int64_t, float, double
		WXF_TOKEN makeNArray() {
			auto num_type = (int)ReadVarint();
			if (num_type > 50) {
				std::cerr << "Unsupported type: " << num_type << std::endl;
				return WXF_TOKEN();
			}

			size_t size_of_type;
			if (num_type >= 16 && num_type < 20) {
				size_of_type = (size_t)1 << (num_type - 16);
			}
			else if (num_type >= 34 && num_type <= 35) {
				size_of_type = (size_t)1 << (num_type - 32);
			}
			else {
				size_of_type = (size_t)1 << num_type;
			}

			WXF_TOKEN node;
			node.type = WXF_HEAD::narray;
			node.rank = (int)ReadVarint();
			node.dimensions = new uint64_t[node.rank + 2];
			size_t all_len = 1;
			for (int i = 0; i < node.rank; i++) {
				node.dimensions[i + 2] = ReadVarint();
				all_len *= node.dimensions[i + 2];
			}
			node.dimensions[0] = num_type;
			node.dimensions[1] = all_len;
			if (num_type >= 16 && num_type < 20) {
				node.u_arr = new uint64_t[all_len];

				std::visit([&](auto&& x) {
					using T = std::decay_t<decltype(x)>;
					T val;
					for (size_t i = 0; i < all_len; i++) {
						std::memcpy(&val, buffer + pos, size_of_type * sizeof(uint8_t));
						node.u_arr[i] = (uint64_t)val;
						pos += size_of_type;
					}
					}
				, select_type(num_type));
			}
			else if (num_type < 10){
				node.i_arr = new int64_t[all_len];

				std::visit([&](auto&& x) {
					using T = std::decay_t<decltype(x)>;
					T val;
					for (size_t i = 0; i < all_len; i++) {
						std::memcpy(&val, buffer + pos, size_of_type * sizeof(uint8_t));
						node.i_arr[i] = (int64_t)val;
						pos += size_of_type;
					}
					}
				, select_type(num_type));
			}
			else if (num_type >= 34 && num_type <= 35) {
				node.d_arr = new double[all_len];

				std::visit([&](auto&& x) {
					using T = std::decay_t<decltype(x)>;
					T val;
					for (size_t i = 0; i < all_len; i++) {
						std::memcpy(&val, buffer + pos, size_of_type * sizeof(uint8_t));
						node.d_arr[i] = (double)val;
						pos += size_of_type;
					}
					}
				, select_type(num_type));
			}

			return node;
		}
	};

	void print_tokens(const std::vector<WXF_TOKEN>& tokens) {
		for (auto& token : tokens) {
			switch (token.type) {
			case WXF_HEAD::i8:
				std::cout << "i8: " << token.i << std::endl;
				break;
			case WXF_HEAD::i16:
				std::cout << "i16: " << token.i << std::endl;
				break;
			case WXF_HEAD::i32:
				std::cout << "i32: " << token.i << std::endl;
				break;
			case WXF_HEAD::i64:
				std::cout << "i64: " << token.i << std::endl;
				break;
			case WXF_HEAD::f64:
				std::cout << "f64: " << token.d << std::endl;
				break;
			case WXF_HEAD::symbol:
				std::cout << "symbol: " << token.str << std::endl;
				break;
			case WXF_HEAD::bigint:
				std::cout << "bigint: " << token.str << std::endl;
				break;
			case WXF_HEAD::bigreal:
				std::cout << "bigreal: " << token.str << std::endl;
				break;
			case WXF_HEAD::string:
				std::cout << "string: " << token.str << std::endl;
				break;
			case WXF_HEAD::binary_string:
				std::cout << "binary_string: " << token.str << std::endl;
				break;
			case WXF_HEAD::func:
				std::cout << "func: " << token.length << " vars" << std::endl;
				break;
			case WXF_HEAD::association:
				std::cout << "association: " << token.length << " rules" << std::endl;
				break;
			case WXF_HEAD::delay_rule:
				std::cout << "delay_rule: " << token.length << std::endl;
				break;
			case WXF_HEAD::rule:
				std::cout << "rule: " << token.length << std::endl;
				break;
			case WXF_HEAD::array: {
				std::cout << "array: rank = " << token.rank << ", dimensions = ";
				size_t all_len = token.dimensions[1];
				for (int i = 0; i < token.rank; i++) {
					std::cout << token.dimensions[i + 2] << " ";
				}
				std::cout << std::endl;

				auto num_type = token.dimensions[0];
				std::cout << "data: ";
				if (num_type < 4) {
					for (int i = 0; i < all_len; i++) {
						std::cout << token.i_arr[i] << " ";
					}
				}
				else if (num_type >= 34 && num_type <= 35) {
					for (int i = 0; i < all_len; i++) {
						std::cout << token.d_arr[i] << " ";
					}
				}
				else {
					std::cerr << "Unknown type" << std::endl;
				}
				std::cout << std::endl;
				break;
			}
			case WXF_HEAD::narray: {
				std::cout << "narray: rank = " << token.rank << ", dimensions = ";
				for (int i = 0; i < token.rank; i++) {
					std::cout << token.dimensions[i + 2] << " ";
				}
				std::cout << std::endl;

				size_t num_type = token.dimensions[0];
				size_t all_len = token.dimensions[1];

				std::cout << "data: ";
				if (num_type >= 16 && num_type < 20) {
					for (size_t i = 0; i < all_len; i++) 
						std::cout << token.u_arr[i] << " ";
				}
				else if (num_type < 4){
					for (size_t i = 0; i < all_len; i++) 
						std::cout << token.i_arr[i] << " ";
				}
				else if (num_type >= 34 && num_type <= 35) {
					for (size_t i = 0; i < all_len; i++) 
						std::cout << token.d_arr[i] << " ";
				}
				else {
					std::cerr << "Unknown type" << std::endl;
				}
				std::cout << std::endl;
				break;
			}
			default:
				std::cerr << "Unknown type" << std::endl;
			}
		}
	}

	/*
		test:
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

		print_tokens(example_test()):
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

	std::vector<WXF_TOKEN> test() {
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
		Parser parser(test);
		parser.parseExpr();

		return std::move(parser.tokens);
	}

} // namespace WXF_PARSER