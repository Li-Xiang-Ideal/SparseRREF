/*
	Copyright (C) 2024 Zhenjie Li (Li, Zhenjie)

	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

#ifndef SPARSE_RREF_H
#define SPARSE_RREF_H

#include "thread_pool.hpp"
#include <algorithm>
#include <bit>
#include <bitset>
#include <charconv> 
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

// flint 
#include "flint/nmod.h"
#include "flint/ulong_extras.h"

#ifdef NULL
#undef NULL
#endif
#define NULL nullptr

namespace SparseRREF {
	enum SPARSE_FILE_TYPE {
		SPARSE_FILE_TYPE_PLAIN,
		SPARSE_FILE_TYPE_SMS,
		SPARSE_FILE_TYPE_MTX,
		SPARSE_FILE_TYPE_BIN
	};

	// Memory management
	template <typename T>
	inline T* s_malloc(const size_t size) {
		return (T*)std::malloc(size * sizeof(T));
	}

	template <typename T>
	inline void s_free(T* s) {
		std::free(s);
	}

	template <typename T>
	inline T* s_realloc(T* s, const size_t size) {
		return (T*)std::realloc(s, size * sizeof(T));
	}

	template <typename T>
	void s_copy(T* des, T* ini, const size_t size) {
		if (des == ini)
			return;
		std::copy(ini, ini + size, des);
	}

	// thread
	using thread_pool = BS::thread_pool<>; // thread pool
	inline size_t thread_id() {
		return BS::this_thread::get_index().value();
	}

	// rref_option
	struct rref_option {
		bool verbose = false;
		bool is_back_sub = true;
		uint8_t method = 0;
		int print_step = 100;
		int search_depth = INT_MAX;
		bool shrink_memory = false;
		std::function<slong(slong)> col_weight = [](slong i) { return i; };
		thread_pool pool = thread_pool(1); // default: thread pool with 1 thread
	};
	using rref_option_t = rref_option[1];

	// version
	constexpr static const char version[] = "v0.3.1";

	inline size_t ctz(ulong x) {
		return std::countr_zero(x);
	}
	inline size_t clz(ulong x) {
		return std::countl_zero(x);
	}
	inline size_t popcount(ulong x) {
		return std::popcount(x);
	}

	// string
	inline void DeleteSpaces(std::string& str) {
		str.erase(std::remove_if(str.begin(), str.end(),
			[](unsigned char x) { return std::isspace(x); }),
			str.end());
	}

	inline std::vector<std::string> SplitString(const std::string& s, const std::string delim) {
		auto start = 0ULL;
		auto end = s.find(delim);
		std::vector<std::string> result;
		while (end != std::string::npos) {
			result.push_back(s.substr(start, end - start));
			start = end + delim.length();
			end = s.find(delim, start);
		}
		result.push_back(s.substr(start, end));
		return result;
	}

	unsigned long long string_to_ull(std::string_view sv) {
		unsigned long long result;
		auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), result);
		if (ec != std::errc()) {
			throw std::runtime_error("Failed to parse number");
		}
		return result;
	}

	// time
	inline std::chrono::system_clock::time_point clocknow() {
		return std::chrono::system_clock::now();
	}

	inline double usedtime(std::chrono::system_clock::time_point start,
		std::chrono::system_clock::time_point end) {
		auto duration =
			std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		return ((double)duration.count() * std::chrono::microseconds::period::num /
			std::chrono::microseconds::period::den);
	}

	// some algorithms
	template <typename T> std::vector<T> difference(std::vector<T> l) {
		std::vector<T> result;
		for (size_t i = 1; i < l.size(); i++) {
			result.push_back(l[i] - l[i - 1]);
		}
		return result;
	}

	template <typename T>
	int lexico_compare(const std::vector<T>& a, const std::vector<T>& b) {
		for (size_t i = 0; i < a.size(); i++) {
			if (a[i] < b[i])
				return -1;
			if (a[i] > b[i])
				return 1;
		}
		return 0;
	}

	template <typename T>
	int lexico_compare(const T* a, const T* b, const size_t len) {
		for (size_t i = 0; i < len; i++) {
			if (a[i] < b[i])
				return -1;
			if (a[i] > b[i])
				return 1;
		}
		return 0;
	}

	template <typename T>
	int lexico_compare(const T* a, const T* b, const std::vector<size_t>& perm) {
		for (auto i : perm) {
			if (a[i] < b[i])
				return -1;
			if (a[i] > b[i])
				return 1;
		}
		return 0;
	}

	// mulit for
	void multi_for(
		const std::vector<size_t>& start, 
		const std::vector<size_t>& end, 
		const std::function<void(std::vector<size_t>&)> func) {

		if (start.size() != end.size()) {
			std::cerr << "Error: start and end size not match." << std::endl;
			exit(1);
		}

		std::vector<size_t> index(start);
		size_t nt = start.size();

		while (true) {
			func(index);

			for (int i = nt - 1; i > -2; i--) {
				if (i == -1) 
					return;
				index[i]++;
				if (index[i] < end[i])
					break;
				index[i] = start[i];
			}
		}
	}

	// uset
	struct uset {
		constexpr static size_t bitset_size = std::numeric_limits<unsigned long long>::digits; // 64 or 32
		std::vector<std::bitset<bitset_size>> data;

		uset() {}

		void resize(size_t alllen) {
			auto len = alllen / bitset_size + 1;
			data.resize(len);
		}

		uset(size_t alllen) {
			resize(alllen);
		}

		~uset() {
			data.clear();
		}

		void insert(size_t val) {
			auto idx = val / bitset_size;
			auto pos = val % bitset_size;
			data[idx].set(pos);
		}

		bool count(size_t val) {
			auto idx = val / bitset_size;
			auto pos = val % bitset_size;
			return data[idx].test(pos);
		}

		void erase(size_t val) {
			auto idx = val / bitset_size;
			auto pos = val % bitset_size;
			data[idx].reset(pos);
		}

		void clear() {
			for (auto& d : data)
				d.reset();
		}

		bool operator[](size_t idx) {
			return count(idx);
		}

		size_t size() {
			return data.size();
		}

		size_t length() {
			return data.size() * bitset_size;
		}

		std::vector<size_t> nonzero() {
			std::vector<ulong> result;
			std::vector<ulong> tmp;
			tmp.reserve(bitset_size);
			for (size_t i = 0; i < data.size(); i++) {
				if (data[i].any()) {
					//// naive version
					// for (size_t j = 0; j < bitset_size; j++) {
					// 	if (data[i].test(j))
					// 		result.push_back(i * bitset_size + j);
					// }

					tmp.clear();
					ulong c = data[i].to_ullong();

					// only ctz version
					// while (c) {
					// 	auto ctzpos = ctz(c);
					// 	result.push_back(i * bitset_size + ctzpos);
					// 	c &= c - 1;
					// }

					while (c) {
						auto ctzpos = ctz(c);
						auto clzpos = bitset_size - 1 - clz(c);
						result.push_back(i * bitset_size + ctzpos);
						if (ctzpos == clzpos)
							break;
						tmp.push_back(i * bitset_size + clzpos);
						c = c ^ (1ULL << clzpos) ^ (1ULL << ctzpos);
					}
					result.insert(result.end(), tmp.rbegin(), tmp.rend());
				}
			}
			return result;
		}
	};

	template <typename T> inline T* binarysearch(T* begin, T* end, T val) {
		auto ptr = std::lower_bound(begin, end, val);
		if (ptr == end || *ptr == val)
			return ptr;
		else
			return end;
	}

	template <typename T> inline T* lower_bound(T* begin, T* end, T* val, ulong rank) {
		if (rank == 1)
			return std::lower_bound(begin, end, *val);

		size_t left = 0;
		size_t right = (end - begin) / rank;

		while (left < right) {
			size_t mid = left + (right - left) / 2;
			if (lexico_compare(begin + rank * mid, val, rank) < 0)
				left = mid + 1;
			else
				right = mid;
		}

		return begin + rank * left;
	}

	template <typename T> inline T* binarysearch(T* begin, T* end, uint16_t rank, T* val) {
		auto ptr = SparseRREF::lower_bound(begin, end, rank, val);
		if (ptr == end || std::equal(ptr, ptr + rank, val))
			return ptr;
		else
			return end;
	}

	template <typename T>
	std::vector<T> perm_init(T n) {
		std::vector<T> perm(n);
		std::iota(perm.begin(), perm.end(), 0);
		return perm;
	}

	std::vector<size_t> perm_inverse(const std::vector<size_t>& perm) {
		size_t n = perm.size();
		std::vector<size_t> result(n);
		for (size_t i = 0; i < n; i++)
			result[perm[i]] = i;
		return result;
	}

	bool is_identity_perm(const std::vector<size_t>& perm) {
		size_t n = perm.size();
		for (size_t i = 0; i < n; i++) {
			if (perm[i] != i)
				return false;
		}
		return true;
	}

	template <typename T>
	void permute(const std::vector<size_t>& Pt, std::vector<T>& A, size_t block_size = 1) {
		size_t n = Pt.size();
		std::vector<bool> visited(n, false);
		auto P = perm_inverse(Pt);
		std::vector<T> temp(block_size);

		for (size_t i = 0; i < n; i++) {
			if (!visited[i]) {
				size_t current = i;
				T* temp = &(A[i * block_size]);

				do {
					int next = P[current];
					for (size_t j = 0; j < block_size; j++)
						std::swap(A[next * block_size + j], temp[j]);
					visited[current] = true;
					current = next;
				} while (current != i);
			}
		}
	}

	template <typename T>
	void permute(const std::vector<size_t>& Pt, T* A, size_t block_size = 1) {
		size_t n = Pt.size();
		std::vector<bool> visited(n, false);
		auto P = perm_inverse(Pt);
		std::vector<T> temp(block_size);

		for (size_t i = 0; i < n; i++) {
			if (!visited[i]) {
				size_t current = i;
				T* temp = &(A[i * block_size]);

				do {
					int next = P[current];
					for (size_t j = 0; j < block_size; j++)
						std::swap(A[next * block_size + j], temp[j]);
					visited[current] = true;
					current = next;
				} while (current != i);
			}
		}
	}

	inline std::vector<size_t> swap_perm(size_t a, size_t b, size_t n) {
		std::vector<size_t> perm(n);
		for (size_t i = 0; i < n; i++)
			perm[i] = i;
		perm[a] = b;
		perm[b] = a;
		return perm;
	}

	// LockFreeQueue
	template <typename T>
	class LockFreeQueue {
	private:
		struct Node {
			std::shared_ptr<T> data;
			std::atomic<Node*> next;

			Node(T const& value) : data(std::make_shared<T>(value)), next(nullptr) {}
		};

		std::atomic<Node*> head;
		std::atomic<Node*> tail;

	public:
		LockFreeQueue() {
			Node* dummy = new Node(T());  
			head.store(dummy);
			tail.store(dummy);
		}

		~LockFreeQueue() {
			while (Node* old_head = head.load()) {
				head.store(old_head->next);
				delete old_head;
			}
		}

		void enqueue(T const& value) {
			Node* new_node = new Node(value);
			Node* old_tail = tail.load();
			Node* null_ptr = nullptr;

			while (true) {
				Node* old_next = old_tail->next.load();

				if (old_next == nullptr) {
					if (old_tail->next.compare_exchange_weak(null_ptr, new_node)) {
						tail.compare_exchange_weak(old_tail, new_node);
						break;
					}
				}
				else {
					tail.compare_exchange_weak(old_tail, old_next);
				}
			}
		}

		std::shared_ptr<T> dequeue() {
			Node* old_head;
			Node* old_tail;
			std::shared_ptr<T> result;

			while (true) {
				old_head = head.load();
				old_tail = tail.load();
				Node* next = old_head->next.load();

				if (old_head == head.load()) {
					if (old_head == old_tail) {
						if (next == nullptr) {
							return std::shared_ptr<T>();  
						}
						tail.compare_exchange_weak(old_tail, next);
					}
					else {
						result = next->data;
						if (head.compare_exchange_weak(old_head, next)) {
							delete old_head;
							return result;
						}
					}
				}
			}
		}
	};

} // namespace SparseRREF

#endif