/*
	Copyright (C) 2025 Zhenjie Li (Li, Zhenjie)

	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/


#ifndef SPARSE_TYPE_H
#define SPARSE_TYPE_H

#include "sparse_rref.h"

namespace SparseRREF {
	enum SPARSE_TYPE {
		SPARSE_CSR, // Compressed sparse row
		SPARSE_COO, // Coordinate list
		SPARSE_LR  // List of rows
	};

	// sparse vector
	template <typename index_type, typename T> struct sparse_vec {
		index_type* indices = NULL;
		T* entries = NULL;
		size_t _nnz = 0;
		size_t _alloc = 0;

		struct de_iterator {
			// use ref if need to modify the value
			index_type& ind;
			T& val;
		};

		struct iterator {
			index_type* ind_ptr;
			T* val_ptr;

			iterator& operator++() { ind_ptr++; val_ptr++; return *this; }
			iterator operator++(int) { iterator tmp = *this; ind_ptr++; val_ptr++; return tmp; }
			iterator& operator--() { ind_ptr--; val_ptr--; return *this; }
			iterator operator--(int) { iterator tmp = *this; ind_ptr--; val_ptr--; return tmp; }
			iterator& operator+=(size_t n) { ind_ptr += n; val_ptr += n; return *this; }
			iterator& operator-=(size_t n) { ind_ptr -= n; val_ptr -= n; return *this; }
			iterator operator+(size_t n) const { iterator tmp = *this; tmp += n; return tmp; }
			iterator operator-(size_t n) const { iterator tmp = *this; tmp -= n; return tmp; }
			bool operator==(const iterator& other) const { return ind_ptr == other.ind_ptr; }
			bool operator!=(const iterator& other) const { return ind_ptr != other.ind_ptr; }

			de_iterator operator*() const { return { *ind_ptr, *val_ptr }; }
		};

		// functions of iterator 
		iterator begin() { return { indices, entries }; }
		iterator end() { return { indices + _nnz, entries + _nnz }; }
		iterator begin() const { return { indices, entries }; }
		iterator end() const { return { indices + _nnz, entries + _nnz }; }
		iterator cbegin() const { return { indices, entries }; }
		iterator cend() const { return { indices + _nnz, entries + _nnz }; }

		auto index_span() const { return std::span<index_type>(indices, _nnz); }
		auto entry_span() const { return std::span<T>(entries, _nnz); }

		// C++23 is needed for zip_view
		// auto index_view() const { return std::ranges::subrange(indices, indices + _nnz); }
		// auto entry_view() const { return std::ranges::subrange(entries, entries + _nnz); }
		// auto combine_view() const { return std::ranges::zip_view(index_view(), entry_view()); }

		sparse_vec() {
			indices = NULL;
			entries = NULL;
			_nnz = 0;
			_alloc = 0;
		}

		void clear() {
			if (_alloc == 0)
				return;
			s_free(indices);
			indices = NULL;
			for (size_t i = 0; i < _alloc; i++)
				entries[i].~T();
			s_free(entries);
			entries = NULL;
			_alloc = 0;
			_nnz = 0;
		}

		~sparse_vec() {
			clear();
		}

		void reserve(size_t n) {
			if (n == _alloc || n == 0)
				return;

			if (_alloc == 0) {
				indices = s_malloc<index_type>(n);
				entries = s_malloc<T>(n);
				for (size_t i = 0; i < n; i++)
					new (entries + i) T();
				_alloc = n;
				return;
			}

			indices = s_realloc(indices, n);

			if (n < _alloc) {
				for (size_t i = n; i < _alloc; i++)
					entries[i].~T();
				entries = s_realloc<T>(entries, n);
			}
			else {
				entries = s_realloc<T>(entries, n);
				for (size_t i = _alloc; i < n; i++)
					new (entries + i) T();
			}

			_alloc = n;
		}

		inline void zero() { _nnz = 0; }
		inline void resize(size_t n) { _nnz = n; }

		inline void copy(const sparse_vec& l) {
			if (this == &l)
				return;
			if (_alloc < l._nnz)
				reserve(l._nnz);
			for (size_t i = 0; i < l._nnz; i++) {
				indices[i] = l.indices[i];
				entries[i] = l.entries[i];
			}
			_nnz = l._nnz;
		}

		inline sparse_vec(const sparse_vec& l) { copy(l); }

		inline size_t nnz() const { return _nnz; }
		inline size_t size() const { return _nnz; }
		inline size_t alloc() const { return _alloc; }

		sparse_vec(sparse_vec&& l) noexcept {
			indices = l.indices;
			entries = l.entries;
			_nnz = l._nnz;
			_alloc = l._alloc;
			l.indices = NULL;
			l.entries = NULL;
			l._nnz = 0;
			l._alloc = 0;
		}

		sparse_vec& operator=(const sparse_vec& l) {
			if (this == &l)
				return *this;

			copy(l);
			return *this;
		}

		sparse_vec& operator=(sparse_vec&& l) noexcept {
			if (this == &l)
				return *this;

			clear();
			indices = l.indices;
			entries = l.entries;
			_nnz = l._nnz;
			_alloc = l._alloc;
			l.indices = NULL;
			l.entries = NULL;
			l._nnz = 0;
			l._alloc = 0;
			return *this;
		}

		inline void push_back(const index_type index, const T& val) {
			if (_nnz + 1 > _alloc)
				reserve((1 + _alloc) * 2); // +1 to avoid _alloc = 0
			indices[_nnz] = index;
			entries[_nnz] = val;
			_nnz++;
		}

		index_type& operator()(const size_t pos) { return indices[pos]; }
		const index_type& operator()(const size_t pos) const { return indices[pos]; }
		T& operator[](const size_t pos) { return entries[pos]; }
		const T& operator[](const size_t pos) const { return entries[pos]; }

		// conversion functions
		template <typename U = T> requires std::is_integral_v<U> || std::is_same_v<U, int_t>
		operator sparse_vec<index_type, rat_t>() {
			sparse_vec<index_type, rat_t> result;
			result.reserve(_nnz);
			result.resize(_nnz);
			for (size_t i = 0; i < _nnz; i++) {
				result.indices[i] = indices[i];
				result.entries[i] = entries[i];
			}
			return result;
		}

		template <typename U = T> requires std::is_integral_v<U>
		operator sparse_vec<index_type, int_t>() {
			sparse_vec<index_type, int_t> result;
			result.reserve(_nnz);
			result.resize(_nnz);
			for (size_t i = 0; i < _nnz; i++) {
				result.indices[i] = indices[i];
				result.entries[i] = entries[i];
			}
			return result;
		}

		template <typename U = T> requires std::is_same_v<U, rat_t>
		sparse_vec<index_type, ulong> operator%(const nmod_t mod) const {
			sparse_vec<index_type, ulong> result;
			result.reserve(_nnz);
			result.resize(_nnz);
			for (size_t i = 0; i < _nnz; i++) {
				result.indices[i] = indices[i];
				result.entries[i] = entries[i] % mod;
			}
			return result;
		}

		void canonicalize() {
			size_t new_nnz = 0;
			for (size_t i = 0; i < _nnz && new_nnz < _nnz; i++) {
				if (entries[i] != 0) {
					if (new_nnz != i) {
						indices[new_nnz] = indices[i];
						entries[new_nnz] = entries[i];
					}
					new_nnz++;
				}
			}
			_nnz = new_nnz;
		}

		void sort_indices() {
			if (_nnz <= 1 || std::is_sorted(indices, indices + _nnz))
				return;

			auto perm = perm_init(_nnz);
			std::sort(perm.begin(), perm.end(), [&](index_type a, index_type b) {
				return indices[a] < indices[b];
				});

			permute(perm, indices);
			permute(perm, entries);
		}

		void compress() {
			canonicalize();
			sort_indices();
			reserve(_nnz);
		}
	};

	template <typename index_type> struct sparse_vec<index_type, bool> {
		index_type* indices = NULL;
		size_t _nnz = 0;
		size_t _alloc = 0;

		auto index_span() const { return std::span<index_type>(indices, _nnz); }

		sparse_vec() {
			indices = NULL;
			_nnz = 0;
			_alloc = 0;
		}

		void clear() {
			if (_alloc != 0)
				s_free(indices);
			_alloc = 0;
			_nnz = 0;
		}

		~sparse_vec() { clear(); }

		void reserve(size_t n) {
			if (n == _alloc)
				return;

			if (_alloc == 0) {
				indices = s_malloc<index_type>(n);
				_alloc = n;
				return;
			}

			indices = s_realloc(indices, n);
			_alloc = n;
		}

		void resize(size_t n) { _nnz = n; }

		inline void copy(const sparse_vec& l) {
			if (this == &l)
				return;
			if (_alloc < l._nnz)
				reserve(l._nnz);
			for (size_t i = 0; i < l._nnz; i++) {
				indices[i] = l.indices[i];
			}
			_nnz = l._nnz;
		}

		sparse_vec(const sparse_vec& l) { copy(l); }
		size_t nnz() const { return _nnz; }

		sparse_vec(sparse_vec&& l) noexcept {
			indices = l.indices;
			_nnz = l._nnz;
			_alloc = l._alloc;
			l.indices = NULL;
			l._nnz = 0;
			l._alloc = 0;
		}

		sparse_vec& operator=(const sparse_vec& l) {
			if (this == &l)
				return *this;

			copy(l);
			return *this;
		}

		sparse_vec& operator=(sparse_vec&& l) noexcept {
			if (this == &l)
				return *this;

			clear();
			indices = l.indices;
			_nnz = l._nnz;
			_alloc = l._alloc;
			l.indices = NULL;
			l._nnz = 0;
			l._alloc = 0;
			return *this;
		}

		void push_back(const index_type index, const bool val = true) {
			if (_nnz + 1 > _alloc)
				reserve((1 + _alloc) * 2); // +1 to avoid _alloc = 0
			indices[_nnz] = index;
			_nnz++;
		}

		index_type& operator()(const size_t pos) { return indices[pos]; }
		const index_type& operator()(const size_t pos) const { return indices[pos]; }
		void zero() { _nnz = 0; }
		void sort_indices() { std::sort(indices, indices + _nnz); }
		void canonicalize() {}
		void compress() { sort_indices(); }
	};

	// new sparse matrix
	template <typename T> struct sparse_mat {
		size_t nrow = 0;
		size_t ncol = 0;
		std::vector<sparse_vec<slong, T>> rows;

		void init(size_t r, size_t c) {
			nrow = r;
			ncol = c;
			rows = std::vector<sparse_vec<slong, T>>(r);
		}

		sparse_mat() { nrow = 0; ncol = 0; }
		~sparse_mat() {}
		sparse_mat(size_t r, size_t c) { init(r, c); }

		sparse_vec<slong, T>& operator[](size_t i) { return rows[i]; }
		const sparse_vec<slong, T>& operator[](size_t i) const { return rows[i]; }

		sparse_mat(const sparse_mat& l) {
			init(l.nrow, l.ncol);
			rows = l.rows;
		}

		sparse_mat(sparse_mat&& l) noexcept {
			nrow = l.nrow;
			ncol = l.ncol;
			rows = std::move(l.rows);
		}

		sparse_mat& operator=(const sparse_mat& l) {
			if (this == &l)
				return *this;
			nrow = l.nrow;
			ncol = l.ncol;
			rows = l.rows;
			return *this;
		}

		sparse_mat& operator=(sparse_mat&& l) noexcept {
			if (this == &l)
				return *this;
			nrow = l.nrow;
			ncol = l.ncol;
			rows = std::move(l.rows);
			return *this;
		}

		void zero() {
			for (size_t i = 0; i < nrow; i++)
				rows[i].zero();
		}

		size_t nnz() const {
			size_t n = 0;
			for (size_t i = 0; i < nrow; i++)
				n += rows[i].nnz();
			return n;
		}

		size_t alloc() const {
			size_t n = 0;
			for (size_t i = 0; i < nrow; i++)
				n += rows[i].alloc();
			return n;
		}

		void compress() {
			for (size_t i = 0; i < nrow; i++) {
				rows[i].compress();
			}
		}

		void clear_zero_row() {
			size_t new_nrow = 0;
			for (size_t i = 0; i < nrow; i++) {
				if (rows[i].nnz() != 0) {
					std::swap(rows[new_nrow], rows[i]);
					new_nrow++;
				}
			}
			nrow = new_nrow;
			rows.resize(nrow);
			rows.shrink_to_fit();
		}

		sparse_mat<T> transpose() {
			sparse_mat<T> res(ncol, nrow);
			for (size_t i = 0; i < nrow; i++)
				res[i].zero();

			for (size_t i = 0; i < nrow; i++) {
				for (size_t j = 0; j < rows[i].nnz(); j++) {
					res[rows[i](j)].push_back(i, rows[i][j]);
				}
			}
			return res;
		}

		template <typename U = T> requires std::is_same_v<U, rat_t>
		sparse_mat<ulong> operator%(const nmod_t mod) const {
			sparse_mat<ulong> result(nrow, ncol);
			for (size_t i = 0; i < nrow; i++) {
				result[i] = rows[i] % mod;
			}
			return result;
		}

		template <typename U = T> requires std::is_same_v<U, rat_t>
		ulong height_bits() const {
			ulong max_height = 0;
			for (size_t i = 0; i < nrow; i++) {
				for (size_t j = 0; j < rows[i].nnz(); j++) {
					auto rr = rows[i][j].height_bits();
					if (rr > max_height)
						max_height = rr;
				}
			}

			return max_height;
		}

		// denominator bits
		template <typename U = T> requires std::is_same_v<U, rat_t>
		ulong den_bits() const {
			int_t den = 1;
			for (size_t i = 0; i < nrow; i++) {
				for (size_t j = 0; j < rows[i].nnz(); j++) {
					if (rows[i][j] != 0 && !(rows[i][j].is_den_one()))
						den = LCM(den, rows[i][j].den());
				}
			}

			return den.bits();
		}
	};

	// CSR format for sparse tensor
	template <typename index_type, typename T> struct sparse_tensor_struct {
		size_t rank;
		size_t alloc;
		index_type* colptr;
		T* valptr;
		std::vector<size_t> dims;
		std::vector<size_t> rowptr;

		using index_v = std::vector<index_type>;
		using index_p = index_type*;

		//empty constructor
		sparse_tensor_struct() {
			rank = 0;
			alloc = 0;
			colptr = NULL;
			valptr = NULL;
		}

		// Constructor with dimensions
		// we require that rank >= 2
		void init(const std::vector<size_t>& l, size_t aoc = 8) {
			dims = l;
			rank = l.size();
			rowptr = std::vector<size_t>(l[0] + 1, 0);
			alloc = aoc;
			colptr = s_malloc<index_type>((rank - 1) * alloc);
			valptr = s_malloc<T>(alloc);
			for (size_t i = 0; i < alloc; i++)
				new (valptr + i) T();
		}

		sparse_tensor_struct(const std::vector<size_t>& l, size_t aoc = 8) {
			init(l, aoc);
		}

		// Copy constructor
		sparse_tensor_struct(const sparse_tensor_struct& l) {
			init(l.dims, l.alloc);
			std::copy(l.rowptr.begin(), l.rowptr.end(), rowptr.begin());
			std::copy(l.colptr, l.colptr + alloc * (rank - 1), colptr);
			for (size_t i = 0; i < alloc; i++)
				valptr[i] = l.valptr[i];
		}

		// Move constructor
		sparse_tensor_struct(sparse_tensor_struct&& l) noexcept {
			dims = l.dims;
			rank = l.rank;
			rowptr = l.rowptr;
			alloc = l.alloc;
			colptr = l.colptr;
			l.colptr = NULL;
			valptr = l.valptr;
			l.valptr = NULL;
			l.alloc = 0; // important for no repeating clear
		}

		void clear() {
			if (alloc == 0)
				return;
			for (size_t i = 0; i < alloc; i++)
				valptr[i].~T();
			s_free(valptr);
			s_free(colptr);
			valptr = NULL;
			colptr = NULL;
			alloc = 0;
		}

		~sparse_tensor_struct() {
			clear();
		}

		void reserve(size_t size) {
			if (size == 0 || size == alloc)
				return;
			if (alloc == 0) {
				alloc = size;
				colptr = s_malloc<index_type>(size * (rank - 1));
				valptr = s_malloc<T>(size);
				for (size_t i = 0; i < size; i++)
					new (valptr + i) T();
				return;
			}
			colptr = s_realloc<index_type>(colptr, size * (rank - 1));
			if (size > alloc) {
				valptr = s_realloc<T>(valptr, size);
				for (size_t i = alloc; i < size; i++)
					new (valptr + i) T();
			}
			else if (size < alloc) {
				for (size_t i = size; i < alloc; i++)
					valptr[i].~T();
				valptr = s_realloc<T>(valptr, size);
			}
			alloc = size;
		}

		void zero() {
			if (rank != 0)
				std::fill(rowptr.begin(), rowptr.end(), 0);
		}

		inline size_t nnz() {
			return rowptr[dims[0]];
		}

		// Copy assignment
		sparse_tensor_struct& operator=(const sparse_tensor_struct& l) {
			if (this == &l)
				return *this;
			auto nz = l.nnz();
			if (alloc == 0) {
				init(l.dims, nz);
				std::copy(l.rowptr.begin(), l.rowptr.end(), rowptr.begin());
				std::copy(l.colptr, l.colptr + nz * (rank - 1), colptr);
				std::copy(l.valptr, l.valptr + nz, valptr);
				return *this;
			}
			dims = l.dims;
			rank = l.rank;
			rowptr = l.rowptr;
			if (alloc < nz)
				reserve(nz);
			std::copy(l.colptr, l.colptr + nz * (rank - 1), colptr);
			std::copy(l.valptr, l.valptr + nz, valptr);
			return *this;
		}

		// Move assignment
		sparse_tensor_struct& operator=(sparse_tensor_struct&& l) noexcept {
			if (this == &l)
				return *this;
			clear();
			dims = l.dims;
			rank = l.rank;
			rowptr = l.rowptr;
			alloc = l.alloc;
			colptr = l.colptr;
			l.colptr = NULL;
			valptr = l.valptr;
			l.valptr = NULL;
			l.alloc = 0; // important for no repeating clear
			return *this;
		}

		std::vector<size_t> row_nums() {
			return SparseRREF::difference(rowptr);
		}

		size_t row_nnz(size_t i) {
			return rowptr[i + 1] - rowptr[i];
		}

		// remove zero entries, double pointer
		void canonicalize() {
			size_t nnz_now = nnz();
			size_t index = 0;
			std::vector<size_t> newrowptr(dims[0] + 1);
			newrowptr[0] = 0;
			for (size_t i = 0; i < dims[0]; i++) {
				for (size_t j = rowptr[i]; j < rowptr[i + 1]; j++) {
					if (valptr[j] != 0) {
						s_copy(colptr + index * (rank - 1), colptr + j * (rank - 1), rank - 1);
						valptr[index] = valptr[j];
						index++;
					}
				}
				newrowptr[i + 1] = index;
			}
			rowptr = newrowptr;
		}

		std::pair<index_p, T*> row(size_t i) {
			return std::make_pair(colptr + rowptr[i] * (rank - 1), valptr + rowptr[i]);
		}

		index_p entry_lower_bound(const index_p l) {
			auto begin = row(l[0]).first;
			auto end = row(l[0] + 1).first;
			if (begin == end)
				return end;
			return SparseRREF::lower_bound(begin, end, l + 1, rank - 1);
		}

		index_p entry_lower_bound(const index_v& l) {
			return entry_lower_bound(l.data());
		}

		index_p entry_ptr(index_p l) {
			auto ptr = entry_lower_bound(l);
			auto end = row(l[0] + 1).first;
			if (ptr == end || std::equal(ptr, ptr + rank - 1, l + 1))
				return ptr;
			else
				return end;
		}

		index_p entry_ptr(const index_v& l) {
			return entry_ptr(l.data());
		}

		// unordered, push back on the end of the row
		void push_back(const index_v& l, const T& val) {
			index_type row = l[0];
			size_t nnz = this->nnz();
			if (nnz + 1 > alloc)
				reserve((alloc + 1) * 2);
			size_t index = rowptr[row + 1];
			for (size_t i = nnz; i > index; i--) {
				auto tmpptr = colptr + (i - 1) * (rank - 1);
				std::copy_backward(tmpptr, tmpptr + (rank - 1), tmpptr + 2 * (rank - 1));
				valptr[i] = valptr[i - 1];
			}
			for (size_t i = 0; i < rank - 1; i++)
				colptr[index * (rank - 1) + i] = l[i + 1];
			valptr[index] = val;
			for (size_t i = row + 1; i <= dims[0]; i++)
				rowptr[i]++;
		}

		// ordered insert
		// mode = false: insert anyway
		// mode = true: insert and replace if exist
		void insert(const index_v& l, const T& val, bool mode = true) {
			size_t trow = l[0];
			size_t nnz = this->nnz();
			if (nnz + 1 > alloc)
				reserve((alloc + 1) * 2);
			auto ptr = entry_lower_bound(l);
			size_t index = (ptr - colptr) / (rank - 1);
			bool exist = (ptr != row(trow + 1).first && std::equal(ptr, ptr + rank - 1, l.data() + 1));
			if (!exist || !mode) {
				for (size_t i = nnz; i > index; i--) {
					auto tmpptr = colptr + (i - 1) * (rank - 1);
					std::copy_backward(tmpptr, tmpptr + (rank - 1), tmpptr + 2 * (rank - 1));
					valptr[i] = valptr[i - 1];
				}
				std::copy(l.begin() + 1, l.begin() + rank, colptr + index * (rank - 1));
				valptr[index] = val;
				for (size_t i = trow + 1; i <= dims[0]; i++)
					rowptr[i]++;
				return;
			}
			valptr[index] = val;
		}

		// ordered add one value
		void insert_add(const index_v& l, const T& val) {
			size_t trow = l[0];
			size_t nnz = this->nnz();
			if (nnz + 1 > alloc)
				reserve((alloc + 1) * 2);
			auto ptr = entry_lower_bound(l);
			size_t index = (ptr - colptr) / (rank - 1);
			bool exist = (ptr != row(trow + 1).first && std::equal(ptr, ptr + rank - 1, l.data() + 1));
			if (!exist) {
				for (size_t i = nnz; i > index; i--) {
					auto tmpptr = colptr + (i - 1) * (rank - 1);
					std::copy_backward(tmpptr, tmpptr + (rank - 1), tmpptr + 2 * (rank - 1));
					valptr[i] = valptr[i - 1];
				}
				std::copy(l.begin() + 1, l.begin() + rank, colptr + index * (rank - 1));
				valptr[index] = val;
				for (size_t i = trow + 1; i <= dims[0]; i++)
					rowptr[i]++;
				return;
			}
			valptr[index] += val;
		}

		sparse_tensor_struct<index_type, T> transpose(const std::vector<size_t>& perm) {
			std::vector<size_t> l(rank);
			std::vector<size_t> lperm(rank);
			for (size_t i = 0; i < rank; i++)
				lperm[i] = dims[perm[i]];
			sparse_tensor_struct<index_type, T> B(lperm, nnz());
			for (size_t i = 0; i < dims[0]; i++) {
				for (size_t j = rowptr[i]; j < rowptr[i + 1]; j++) {
					l[0] = i;
					auto tmpptr = colptr + j * (rank - 1);
					for (size_t k = 1; k < rank; k++)
						l[k] = tmpptr[k - 1];
					for (size_t k = 0; k < rank; k++)
						lperm[k] = l[perm[k]];
					B.push_back(lperm, valptr[j]);
				}
			}
			return B;
		}

		void sort_indices() {
			for (size_t i = 0; i < dims[0]; i++) {
				size_t rownnz = rowptr[i + 1] - rowptr[i];
				std::vector<size_t> perm(rownnz);
				for (size_t j = 0; j < rownnz; j++)
					perm[j] = j;
				std::sort(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
					auto ptra = colptr + (rowptr[i] + a) * (rank - 1);
					auto ptrb = colptr + (rowptr[i] + b) * (rank - 1);
					return lexico_compare(ptra, ptrb, rank - 1) < 0;
					});

				permute(perm, colptr + rowptr[i] * (rank - 1), rank - 1);
				permute(perm, valptr + rowptr[i]);
			}
		}
	};

	// define the default sparse tensor
	template <typename index_type, typename T, SPARSE_TYPE Type = SPARSE_COO> struct sparse_tensor;

	template <typename index_type, typename T> struct sparse_tensor<index_type, T, SPARSE_CSR> {
		sparse_tensor_struct<index_type, T> data;

		using index_v = std::vector<index_type>;
		using index_p = index_type*;

		void clear() { data.clear(); }

		sparse_tensor() {}
		~sparse_tensor() {}
		sparse_tensor(std::vector<size_t> l, size_t aoc = 8) : data(l, aoc) {}
		sparse_tensor(const sparse_tensor& l) : data(l.data) {}
		sparse_tensor(sparse_tensor&& l) noexcept : data(std::move(l.data)) {}
		sparse_tensor& operator=(const sparse_tensor& l) { data = l.data; return *this; }
		sparse_tensor& operator=(sparse_tensor&& l) noexcept { data = std::move(l.data); return *this; }

		inline size_t alloc() const { return data.alloc; }
		inline size_t rank() const { return data.rank; }
		inline size_t nnz() const { return data.rowptr[data.dims[0]]; }
		inline std::vector<size_t> dims() const { return data.dims; }
		inline size_t dim(size_t i) const { return data.dims[i]; }
		inline void zero() { data.zero(); }
		inline void insert(const index_v& l, const T& val, bool mode = true) { data.insert(l, val, mode); }
		inline void push_back(const index_v& l, const T& val) { data.push_back(l, val); }
		inline void canonicalize() { data.canonicalize(); }
		inline void sort_indices() { data.sort_indices(); }
		inline sparse_tensor transpose(const std::vector<size_t>& perm) {
			sparse_tensor B;
			B.data = data.transpose(perm);
			return B;
		}

		void convert_from_COO(const sparse_tensor<index_type, T, SPARSE_COO>& l) {
			std::vector<size_t> dims(l.data.dims.begin() + 1, l.data.dims.end()); // remove the first dimension
			size_t nnz = l.data.rowptr[1];
			size_t rank = dims.size();
			data.init(dims, nnz);
			std::vector<index_type> index(rank);
			for (size_t i = 0; i < nnz; i++) {
				for (size_t j = 0; j < rank; j++)
					index[j] = l.data.colptr[i * rank + j];
				data.push_back(index, l.data.valptr[i]);
			}
		}

		// constructor from COO
		sparse_tensor(const sparse_tensor<index_type, T, SPARSE_COO>& l) { convert_from_COO(l); }
		sparse_tensor& operator=(const sparse_tensor<index_type, T, SPARSE_COO>& l) {
			data.clear();
			convert_from_COO(l);
			return *this;
		}

		// only for test
		void print_test() {
			for (size_t i = 0; i < data.dims[0]; i++) {
				for (size_t j = data.rowptr[i]; j < data.rowptr[i + 1]; j++) {
					std::cout << i << " ";
					for (size_t k = 0; k < data.rank - 1; k++)
						std::cout << (size_t)data.colptr[j * (data.rank - 1) + k] << " ";
					std::cout << " : " << data.valptr[j] << std::endl;
				}
			}
		}
	};

	template <typename index_type, typename T> struct sparse_tensor<index_type, T, SPARSE_COO> {
		sparse_tensor_struct<index_type, T> data;

		using index_v = std::vector<index_type>;
		using index_p = index_type*;

		template <typename S>
		std::vector<S> prepend_num(const std::vector<S>& l, S num = 0) {
			std::vector<S> lp;
			lp.reserve(l.size() + 1);
			lp.push_back(num);
			lp.insert(lp.end(), l.begin(), l.end());
			return lp;
		}

		void clear() { data.clear(); }
		void init(const std::vector<size_t>& l, size_t aoc = 8) {
			data.init(prepend_num(l, (size_t)1), aoc);
		}

		sparse_tensor() {}
		~sparse_tensor() {}
		sparse_tensor(const std::vector<size_t>& l, size_t aoc = 8) : data(prepend_num(l, (size_t)1), aoc) {}
		sparse_tensor(const sparse_tensor& l) : data(l.data) {}
		sparse_tensor(sparse_tensor&& l) noexcept : data(std::move(l.data)) {}
		sparse_tensor& operator=(const sparse_tensor& l) { data = l.data; return *this; }
		sparse_tensor& operator=(sparse_tensor&& l) noexcept { data = std::move(l.data); return *this; }

		// for the i-th column, return the indices
		index_p index(size_t i) const { return data.colptr + i * rank(); }
		T& val(size_t i) const { return data.valptr[i]; }

		index_v index_vector(size_t i) const {
			index_v result(rank());
			for (size_t j = 0; j < rank(); j++)
				result[j] = index(i)[j];
			return result;
		}

		inline size_t alloc() const { return data.alloc; }
		inline size_t nnz() const { return data.rowptr[1]; }
		inline size_t rank() const { return data.rank - 1; }
		inline std::vector<size_t> dims() const {
			std::vector<size_t> result(data.dims.begin() + 1, data.dims.end());
			return result;
		}
		inline size_t dim(size_t i) const { return data.dims[i + 1]; }
		inline void zero() { data.zero(); }
		inline void reserve(size_t size) { data.reserve(size); }
		inline void resize(size_t new_nnz) {
			if (new_nnz > alloc())
				reserve(new_nnz);
			data.rowptr[1] = new_nnz;
		}

		// change the dimensions of the tensor
		// it is dangerous, only for internal use
		inline void change_dims(const std::vector<size_t>& new_dims) {
			auto dims = prepend_num(new_dims, (size_t)1);
			data.dims = dims;
			data.rank = dims.size();
			data.colptr = s_realloc<index_type>(data.colptr, new_dims.size() * alloc());
		}

		inline void flatten(const std::vector<std::vector<size_t>>& pos) {
			auto r = rank();
			auto nr = pos.size();
			std::vector<index_type> newindex(nr);
			std::vector<size_t> new_dims(nr);
			auto old_dim = dims();
			auto init_ptr = data.colptr;

			// first compute new dimensions
			for (size_t i = 0; i < nr; i++) {
				new_dims[i] = 1;
				for (auto j : pos[i])
					new_dims[i] *= old_dim[j];
			}
			new_dims = prepend_num(new_dims, (size_t)1);

			for (size_t i = 0; i < nnz(); i++) {
				auto ptr = index(i);
				for (size_t j = 0; j < nr; j++) {
					newindex[j] = 0;
					for (auto k : pos[j])
						newindex[j] = newindex[j] * old_dim[k] + ptr[k];
				}
				for (size_t j = 0; j < nr; j++)
					init_ptr[i * nr + j] = newindex[j];
			}
			s_realloc(data.colptr, nr * nnz());

			// change the dimensions
			data.dims = new_dims;
			data.rank = nr;
		}

		// TODO: resharp, for example {2,100} to {2,5,20}
		inline void resharp() {}

		inline void insert(const index_v& l, const T& val, bool mode = true) { data.insert(prepend_num(l), val, mode); }
		inline void insert_add(const index_v& l, const T& val) { data.insert_add(prepend_num(l), val); }
		void push_back(const index_p l, const T& new_val) {
			auto n_nnz = nnz();
			if (n_nnz + 1 > data.alloc)
				reserve((data.alloc + 1) * 2);
			s_copy(index(n_nnz), l, rank());
			val(n_nnz) = new_val;
			data.rowptr[1]++; // increase the nnz
		}
		void push_back(const index_v& l, const T& new_val) {
			auto n_nnz = nnz();
			if (n_nnz + 1 > data.alloc)
				reserve((data.alloc + 1) * 2);
			std::copy(l.begin(), l.end(), index(n_nnz));
			val(n_nnz) = new_val;
			data.rowptr[1]++; // increase the nnz
		}
		inline void canonicalize() { data.canonicalize(); }
		inline void sort_indices() { data.sort_indices(); }
		inline sparse_tensor transpose(const std::vector<size_t>& perm) {
			std::vector<size_t> perm_new(perm);
			for (auto& a : perm_new) { a++; }
			perm_new = prepend_num(perm_new, 0);
			sparse_tensor B;
			B.data = data.transpose(perm_new);
			B.sort_indices();
			return B;
		}

		std::vector<size_t> gen_perm() const {
			std::vector<size_t> perm = perm_init(nnz());

			auto r = rank();
			std::sort(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
				return lexico_compare(index(a), index(b), r) < 0;
				});
			return perm;
		}

		std::vector<size_t> gen_perm(const std::vector<size_t>& index_perm) const {
			if (index_perm.size() != rank()) {
				std::cerr << "Error: gen_perm: index_perm size is not equal to rank" << std::endl;
				exit(1);
			}

			std::vector<size_t> perm = perm_init(nnz());
			std::sort(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
				return lexico_compare(index(a), index(b), index_perm) < 0;
				});

			return perm;
		}

		void transpose_replace(const std::vector<size_t>& perm, thread_pool* pool = nullptr) {
			std::vector<size_t> new_dims(rank() + 1);
			new_dims[0] = data.dims[0];

			for (size_t i = 0; i < rank(); i++)
				new_dims[i + 1] = data.dims[perm[i] + 1];
			data.dims = new_dims;

			if (pool == nullptr) {
				std::vector<size_t> index_new(rank());
				for (size_t i = 0; i < nnz(); i++) {
					auto ptr = index(i);
					for (size_t j = 0; j < rank(); j++)
						index_new[j] = ptr[perm[j]];
					std::copy(index_new.begin(), index_new.end(), ptr);
				}
			}
			else {
				pool->detach_blocks(0, nnz(), [&](size_t ss, size_t ee) {
					std::vector<size_t> index_new(rank());
					for (size_t i = ss; i < ee; i++) {
						auto ptr = index(i);
						for (size_t j = 0; j < rank(); j++)
							index_new[j] = ptr[perm[j]];
						std::copy(index_new.begin(), index_new.end(), ptr);
					}
					});
				pool->wait();
			}
		}

		sparse_tensor<index_type, T, SPARSE_COO> chop(slong pos, slong aa) const {
			std::vector<size_t> dims_new = dims();
			dims_new.erase(dims_new.begin() + pos);
			sparse_tensor<index_type, T, SPARSE_COO> result(dims_new);
			index_v index_new;
			index_new.reserve(rank() - 1);
			for (size_t i = 0; i < nnz(); i++) {
				if (index(i)[pos] != aa)
					continue;
				for (size_t j = 0; j < rank(); j++) {
					if (j != pos)
						index_new.push_back(index(i)[j]);
				}
				result.push_back(index_new, val(i));
				index_new.clear();
			}
			return result;
		}

		// constructor from CSR
		sparse_tensor(const sparse_tensor<index_type, T, SPARSE_CSR>& l) {
			data.init(prepend_num(l.dims(), (size_t)1), l.nnz());
			resize(l.nnz());

			auto r = rank();
			auto n_row = dim(0);

			// first copy the data
			s_copy(data.valptr, l.data.valptr, l.nnz());

			// then deal with the indices
			for (size_t i = 0; i < n_row; i++) {
				for (size_t j = l.data.rowptr[i]; j < l.data.rowptr[i + 1]; j++) {
					auto tmp_index = index(j);
					tmp_index[0] = i;
					for (size_t k = 0; k < r - 1; k++)
						tmp_index[k + 1] = l.data.colptr[j * (r - 1) + k];
				}
			}
		}

		sparse_tensor& operator=(const sparse_tensor<index_type, T, SPARSE_CSR>& l) {
			if (alloc() == 0) {
				init(l.dims(), l.nnz());
			}
			else {
				change_dims(l.dims());
				reserve(l.nnz());
			}

			auto r = rank();
			auto n_row = dim(0);

			// first copy the data
			s_copy(data.valptr, l.data.valptr, l.nnz());

			// then deal with the indices
			for (size_t i = 0; i < n_row; i++) {
				for (size_t j = l.data.rowptr[i]; j < l.data.rowptr[i + 1]; j++) {
					auto tmp_index = index(j);
					tmp_index[0] = i;
					for (size_t k = 0; k < r - 1; k++)
						tmp_index[k + 1] = l.data.colptr[j * (r - 1) + k];
				}
			}

			return *this;
		}

		sparse_tensor& operator=(sparse_tensor<index_type, T, SPARSE_CSR>&& l) {
			data.rank = l.rank() + 1;
			data.alloc = l.data.alloc;
			data.colptr = s_realloc(data.colptr, l.data.alloc * l.rank());
			std::swap(data.valptr, l.data.valptr); // no need to copy
			data.dims = prepend_num(l.dims(), (size_t)1);
			data.rowptr = { 0, l.nnz() };

			auto r = l.rank();
			auto n_row = dim(0);

			// then deal with the indices
			for (size_t i = 0; i < n_row; i++) {
				for (size_t j = l.data.rowptr[i]; j < l.data.rowptr[i + 1]; j++) {
					auto tmp_index = index(j);
					tmp_index[0] = i;
					for (size_t k = 0; k < r - 1; k++)
						tmp_index[k + 1] = l.data.colptr[j * (r - 1) + k];
				}
			}

			return *this;
		}

		sparse_tensor(sparse_tensor<index_type, T, SPARSE_CSR>&& l) noexcept {
			data.rank = l.rank() + 1;
			data.alloc = l.data.alloc;
			data.colptr = s_malloc<index_type>(l.data.alloc * l.rank());
			data.valptr = l.data.valptr;
			l.data.valptr = NULL;
			data.dims = prepend_num(l.dims(), (size_t)1);
			data.rowptr = { 0, l.nnz() };

			auto r = l.rank();
			auto n_row = dim(0);

			// then deal with the indices
			for (size_t i = 0; i < n_row; i++) {
				for (size_t j = l.data.rowptr[i]; j < l.data.rowptr[i + 1]; j++) {
					auto tmp_index = index(j);
					tmp_index[0] = i;
					for (size_t k = 0; k < r - 1; k++)
						tmp_index[k + 1] = l.data.colptr[j * (r - 1) + k];
				}
			}
		}

		void print_test() {
			for (size_t j = 0; j < data.rowptr[1]; j++) {
				for (size_t k = 0; k < data.rank - 1; k++)
					std::cout << (size_t)(data.colptr[j * (data.rank - 1) + k]) << " ";
				std::cout << " : " << data.valptr[j] << std::endl;
			}
		}
	};
}

#endif