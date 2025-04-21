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
	entries, so some types are not supported, such as complex numbers, strings, etc.
	The full list of supported types is given below:

	done	byte value  type of part
	*		102			function
	*		67			int8_t
	*		106			int16_t
	*		105			int32_t
	*		76			int64_t
			114			machine reals
			83			string
			66			binary string
	*		115			symbol
	*		73			big integer
			82			big real
	*		193			packed array
			194			numeric array
			65			association
			58			delayed rule in association
			45			rule in association

*/


#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <cstring>

namespace WXF_PARSER {

	enum WXF_HEAD {
		func = 102,
		i8 = 67,
		i16 = 106,
		i32 = 105,
		i64 = 76,
		symbol = 115,
		array = 193,
		bigint = 73
	};

	struct WXF_TOKEN {
		WXF_HEAD type;
		int rank;
		union { // length of the data
			uint64_t length;
			uint64_t* dimensions; // for array
		};
		union { // data
			int64_t i;
			double d; // for f64
			int64_t* i_arr; // for array
			char* str; // for symbol and bigint
			// TODO: add more types
		};

		WXF_TOKEN() : type(WXF_HEAD::i8), rank(0), length(0), i(0) {}

		void clear() {
			if (type == WXF_HEAD::symbol || type == WXF_HEAD::bigint) {
				delete[] str;
			}
			else if (type == WXF_HEAD::array) {
				delete[] dimensions;
				delete[] i_arr;
			}
			// no need to clear i, length, rank, type, as they are just basic types
		}

		~WXF_TOKEN() { clear(); }

		// disable copy constructor and copy assignment operator
		WXF_TOKEN(const WXF_TOKEN&) = delete; 
		WXF_TOKEN& operator=(const WXF_TOKEN&) = delete; 

		// move constructor
		WXF_TOKEN(WXF_TOKEN&& other) noexcept : type(other.type), rank(other.rank), length(other.length), i(other.i) {
			if (type == WXF_HEAD::symbol || type == WXF_HEAD::bigint) {
				str = other.str;
				other.str = nullptr;
			}
			else if (type == WXF_HEAD::array) {
				dimensions = other.dimensions;
				i_arr = other.i_arr;
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
		// so the maximal length is 10 ( 7*9+8>64 )
		uint64_t ReadVarint() {
			size_t count = 0;
			uint64_t result = 0;
			auto ptr = buffer + pos;

			while (pos < size) {
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
					case WXF_HEAD::symbol: 
						tokens.push_back(makeSymbol());
						break;
					case WXF_HEAD::func: 
						tokens.push_back(makeFunction());
						break;
					case WXF_HEAD::array: 
						tokens.push_back(makeArray());
						break;
					case WXF_HEAD::bigint: 
						tokens.push_back(makeBigInt());
						break;
					default:
						std::cerr << "Unknown head type: " << (int)type << " pos: " << pos << std::endl;
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
			
			node.i = (int64_t)val;
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
			return node;
		}

		// symbol, length, str
		WXF_TOKEN makeSymbol() {
			WXF_TOKEN node;
			node.type = WXF_HEAD::symbol;
			node.length = ReadVarint();
			node.str = new char[node.length + 1];
			std::memcpy(node.str, buffer + pos, node.length);
			node.str[node.length] = '\0'; // add null terminator
			pos += node.length;

			return node;
		}

		// bigint, length, str
		WXF_TOKEN makeBigInt() {
			WXF_TOKEN node;
			node.type = WXF_HEAD::bigint;
			node.length = ReadVarint();
			node.str = new char[node.length + 1];
			std::memcpy(node.str, buffer + pos, node.length);
			node.str[node.length] = '\0'; // add null terminator
			pos += node.length;

			return node;
		}

		// func, lenth
		WXF_TOKEN makeFunction() {
			WXF_TOKEN node;
			node.type = WXF_HEAD::func;
			node.length = ReadVarint();
			return node;
		}

		// array, rank, dimensions, data
		WXF_TOKEN makeArray() {
			auto num_type = ReadVarint();
			// 0 is int8_t      1 is int16_t
			// 2 is int32_t     3 is int64_t
			// 34 float         35 double
			// 51 complex float 52 complex double
			// we only support int8_t, int16_t, int32_t, int64_t
			if (num_type > 3) {
				std::cerr << "Unsupported type: " << num_type << std::endl;
				return WXF_TOKEN();
			}
			size_t size_of_type = (size_t)1 << num_type;

			WXF_TOKEN node;
			node.type = WXF_HEAD::array;
			node.rank = (int)ReadVarint();
			node.dimensions = new uint64_t[node.rank];
			size_t all_len = 1;
			for (int i = 0; i < node.rank; i++) {
				node.dimensions[i] = ReadVarint();
				all_len *= node.dimensions[i];
			}
			node.i_arr = new int64_t[all_len];
			for (size_t i = 0; i < all_len; i++) {
				std::memcpy(node.i_arr + i, buffer + pos, size_of_type * sizeof(uint8_t));
				pos += size_of_type;
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
			case WXF_HEAD::symbol:
				std::cout << "symbol: " << token.str << std::endl;
				break;
			case WXF_HEAD::bigint:
				std::cout << "bigint: " << token.str << std::endl;
				break;
			case WXF_HEAD::func:
				std::cout << "func: " << token.length << " vars" << std::endl;
				break;
			case WXF_HEAD::array: {
				std::cout << "array: rank = " << token.rank << ", dimensions = ";
				size_t all_len = 1;
				for (int i = 0; i < token.rank; i++) {
					std::cout << token.dimensions[i] << " ";
					all_len *= token.dimensions[i];
				}
				std::cout << std::endl;
				std::cout << "data: ";
				for (int i = 0; i < all_len; i++) {
					std::cout << token.i_arr[i] << " ";
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
		example test:
			SparseArray[Automatic,List[2,3],0,List[1,List[List[0,2,3],
			List[List[1],List[3],List[2]]],
			List[Rational[1,3],3,Rational[-4,33333333333333444333333335]]]]
	*/

	std::vector<WXF_TOKEN> example_test() {
		std::vector<uint8_t> test{ 56, 58, 102, 4, 115, 11, 83, 112, 97, 114, 115, 101, 65, 114, 114, \
								97, 121, 115, 9, 65, 117, 116, 111, 109, 97, 116, 105, 99, 193, 0, 1, \
								2, 2, 3, 67, 0, 102, 3, 115, 4, 76, 105, 115, 116, 67, 1, 102, 2, \
								115, 4, 76, 105, 115, 116, 193, 0, 1, 3, 0, 2, 3, 193, 0, 2, 3, 1, 1, \
								3, 2, 102, 3, 115, 4, 76, 105, 115, 116, 102, 2, 115, 8, 82, 97, 116, \
								105, 111, 110, 97, 108, 67, 1, 67, 3, 67, 3, 102, 2, 115, 8, 82, 97, \
								116, 105, 111, 110, 97, 108, 67, 252, 73, 26, 51, 51, 51, 51, 51, 51, \
								51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 51, 51, 51, 51, 51, 51, \
								51, 51, 53 };
		Parser parser(test);
		parser.parseExpr();

		return std::move(parser.tokens);
	}

} // namespace WXF_PARSER