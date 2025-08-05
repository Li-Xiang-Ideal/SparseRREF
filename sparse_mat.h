/*
	Copyright (C) 2024 Zhenjie Li (Li, Zhenjie)

	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/


#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#include "sparse_vec.h"
#include "wxf_support.h"

namespace SparseRREF {
	template <typename T, typename index_t>
	inline T* sparse_mat_entry(sparse_mat<T, index_t>& mat, size_t r, index_t c, bool isbinary = true) {
		return sparse_vec_entry(mat[r], c, isbinary);
	}

	template <typename T, typename S, typename index_t>
	void sparse_mat_transpose_part_replace(
		sparse_mat<S, index_t>& tranmat, const sparse_mat<T, index_t>& mat,
		const std::vector<index_t>& rows, thread_pool* pool = nullptr) {
		tranmat.zero();

		if (pool == nullptr) {
			for (size_t i = 0; i < rows.size(); i++) {
				for (size_t j = 0; j < mat[rows[i]].nnz(); j++) {
					auto col = mat[rows[i]](j);
					if constexpr (std::is_same_v<S, T>) {
						tranmat[col].push_back(rows[i], mat[rows[i]][j]);
					}
					else if constexpr (std::is_same_v<S, bool>) {
						tranmat[col].push_back(rows[i], true);
					}
				}
			}
			return;
		}
		
		size_t mtx_size;
		if (pool->get_thread_count() > 16)
			mtx_size = 65536;
		else
			mtx_size = 1 << pool->get_thread_count();
		std::vector<std::mutex> mtxes(mtx_size);

		pool->detach_loop(0, rows.size(), [&](size_t i) {
			for (size_t j = 0; j < mat[rows[i]].nnz(); j++) {
				auto col = mat[rows[i]](j);
				std::lock_guard<std::mutex> lock(mtxes[col % mtx_size]);
				if constexpr (std::is_same_v<S, T>) {
					tranmat[col].push_back(rows[i], mat[rows[i]][j]);
				}
				else if constexpr (std::is_same_v<S, bool>) {
					tranmat[col].push_back(rows[i], true);
				}
			}
			});
		pool->wait();
	}

	template <typename T, typename S, typename index_t>
	void sparse_mat_transpose_replace(sparse_mat<S, index_t>& tranmat,
		const sparse_mat<T, index_t>& mat,
		thread_pool* pool = nullptr) {
		std::vector<index_t> rows(mat.nrow);
		for (size_t i = 0; i < mat.nrow; i++)
			rows[i] = i;
		sparse_mat_transpose_part_replace(tranmat, mat, rows, pool);
	}

	// join two sparse matrices
	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_join(const sparse_mat<T, index_t>& A, const sparse_mat<T, index_t>& B, thread_pool* pool = nullptr) {
		if (A.ncol != B.ncol)
			throw std::invalid_argument("sparse_mat_join: column number mismatch");

		sparse_mat<T, index_t> res(A.nrow + B.nrow, A.ncol);

		if (pool == nullptr) {
			std::copy(A.rows.begin(), A.rows.end(), res.rows.begin());
			std::copy(B.rows.begin(), B.rows.end(), res.rows.begin() + A.nrow);
		}
		else {
			pool->detach_loop(0, A.nrow, [&](size_t i) {
				res[i] = A[i];
			});
			pool->wait();
			pool->detach_loop(0, B.nrow, [&](size_t i) {
				res[i + A.nrow] = B[i];
			});
			pool->wait();
		}

		return res;
	}

	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_join(sparse_mat<T, index_t>&& A, sparse_mat<T, index_t>&& B) {
		if (A.ncol != B.ncol)
			throw std::invalid_argument("sparse_mat_join: column number mismatch");

		sparse_mat<T, index_t> res = std::move(A);
		res.append(std::move(B));
		return res;
	}

	// split a sparse matrix into two parts
	template <typename T, typename index_t>
	std::pair<sparse_mat<T, index_t>, sparse_mat<T, index_t>> sparse_mat_split(const sparse_mat<T, index_t>& mat, const size_t split_row, thread_pool* pool = nullptr) {
		if (split_row > mat.nrow)
			throw std::out_of_range("sparse_mat_split: split_row out of range");

		sparse_mat<T, index_t> A(split_row, mat.ncol);
		sparse_mat<T, index_t> B(mat.nrow - split_row, mat.ncol);
		
		if (pool == nullptr) {
			std::copy(mat.rows.begin(), mat.rows.begin() + split_row, A.rows.begin());
			std::copy(mat.rows.begin() + split_row, mat.rows.end(), B.rows.begin());
		}
		else {
			pool->detach_loop(0, split_row, [&](size_t i) {
				A[i] = mat[i];
			});
			pool->wait();
			pool->detach_loop(split_row, mat.nrow, [&](size_t i) {
				B[i - split_row] = mat[i];
			});
			pool->wait();
		}

		return {A, B};
	}

	template <typename T, typename index_t>
	std::pair<sparse_mat<T, index_t>, sparse_mat<T, index_t>> sparse_mat_split(sparse_mat<T, index_t>&& mat, const size_t split_row) {
		if (split_row > mat.nrow)
			throw std::out_of_range("sparse_mat_split: split_row out of range");

		sparse_mat<T, index_t> A, B;
		A.nrow = split_row;
		A.ncol = mat.ncol;
		B.nrow = mat.nrow - split_row;
		B.ncol = mat.ncol;

		A.rows = std::vector<sparse_vec<T, index_t>>(std::make_move_iterator(mat.rows.begin()),
			std::make_move_iterator(mat.rows.begin() + split_row));
		B.rows = std::vector<sparse_vec<T, index_t>>(std::make_move_iterator(mat.rows.begin() + split_row),
			std::make_move_iterator(mat.rows.end()));

		mat.rows.clear();
		mat.rows.shrink_to_fit();
		mat.nrow = 0;
		mat.ncol = 0;

		return {std::move(A), std::move(B)};
	}

	// rref staffs

	// first look for rows with only one nonzero value and eliminate them
	// we assume that mat is canonical, i.e. each index is sorted
	// and the result is also canonical
	template <typename T, typename index_t>
	size_t eliminate_row_with_one_nnz(sparse_mat<T, index_t>& mat, std::vector<index_t>& donelist,
		rref_option_t opt) {
		auto localcounter = 0;
		std::unordered_map<index_t, index_t> pivlist;
		bit_array collist(mat.ncol);
		for (size_t i = 0; i < mat.nrow; i++) {
			if (donelist[i] != -1)
				continue;
			if (mat[i].nnz() == 1) {
				if (!collist[mat[i](0)]) {
					localcounter++;
					pivlist[i] = mat[i](0);
					collist.insert(mat[i](0));
				}
			}
		}

		if (localcounter == 0)
			return localcounter;

		opt->pool.detach_loop(0, mat.nrow, [&](size_t i) {
			bool is_changed = false;
			for (auto [col, val] : mat[i]) {
				if (collist[col]) {
					if (pivlist.contains(i) && pivlist[i] == col)
						val = 1;
					else {
						val = 0;
						is_changed = true;
					}
				}
			}
			if (is_changed) {
				mat[i].canonicalize();
				mat[i].reserve(mat[i].nnz());
			}
			});

		for (auto [a, b] : pivlist)
			donelist[a] = b;

		opt->pool.wait();

		return localcounter;
	}

	template <typename T, typename index_t>
	size_t eliminate_row_with_one_nnz_rec(sparse_mat<T, index_t>& mat, std::vector<index_t>& donelist,
		rref_option_t opt, int max_depth = 1024) {
		int depth = 0;
		size_t localcounter = 0;
		size_t count = 0;
		bool verbose = opt->verbose;
		bool dir = true;

		std::string dirstr = (dir) ? "Col" : "Row";
		size_t ndir = (dir) ? mat.ncol : mat.nrow;

		size_t oldnnz = mat.nnz();
		int bitlen_nnz = (int)std::floor(std::log(oldnnz) / std::log(10)) + 2;
		int bitlen_ndir = (int)std::floor(std::log(ndir) / std::log(10)) + 1;

		// if the number of newly eliminated rows is less than 
		// 0.1% of the total number of eliminated rows, we stop
		do {
			localcounter = eliminate_row_with_one_nnz(mat, donelist, opt);
			count += localcounter;
			if (verbose) {
				oldnnz = mat.nnz();
				std::cout << "-- " << dirstr << ": " << std::setw(bitlen_ndir)
					<< count << "/" << ndir
					<< "  rank: " << std::setw(bitlen_ndir) << count
					<< "  nnz: " << std::setw(bitlen_nnz) << oldnnz
					<< "  density: " << std::setprecision(6) << std::setw(8)
					<< 100 * (double)oldnnz / (mat.nrow * mat.ncol) << "%"
					<< "    \r" << std::flush;
			}
			depth++;
			if (opt->abort)
				return count;

		} while (localcounter > 0 && depth < max_depth && localcounter > 10);
		return count;
	}

	// first choose the pivot with minimal col_weight
	// if the col_weight is negative, we do not choose it
	template <typename T, typename index_t>
	std::vector<pivot_t<index_t>> pivots_search(
		const sparse_mat<T, index_t>& mat, const sparse_mat<bool, index_t>& tranmat,
		const std::vector<index_t>& rowpivs, const std::vector<index_t>& leftrows,
		const std::vector<index_t>& leftcols,
		const std::function<int64_t(int64_t)>& col_weight = [](int64_t i) { return i; }) {

		std::list<pivot_t<index_t>> pivots;
		std::unordered_set<index_t> dict;
		dict.reserve((size_t)4096);

		// leftlook first
		for (auto col : leftcols) {
			if (tranmat[col].nnz() == 0)
				continue;
			// negative weight means that we do not want to select this column
			if (col_weight(col) < 0)
				continue;

			index_t row;
			size_t mnnz = SIZE_MAX;
			bool flag = true;

			for (auto r : tranmat[col].index_span()) {
				flag = (dict.count(r) == 0);
				if (!flag)
					break;
				if (rowpivs[r] != -1)
					continue;
				size_t newnnz = mat[r].nnz();
				if (newnnz < mnnz) {
					row = r;
					mnnz = newnnz;
				}
				// make the result stable
				else if (newnnz == mnnz) {
					if (r < row)
						row = r;
				}
			}
			if (!flag)
				continue;
			if (mnnz != SIZE_MAX) {
				pivots.emplace_front(row, col);
				dict.insert(row);
			}
		}

		// rightlook then
		dict.clear();

		for (auto [r, c] : pivots)
			dict.insert(c);

		for (auto row : leftrows) {
			index_t col = 0;
			size_t mnnz = SIZE_MAX;
			bool flag = true;

			for (auto c : mat[row].index_span()) {
				// negative weight means that we do not want to select this column
				if (col_weight(c) < 0)
					continue;
				flag = (dict.count(c) == 0);
				if (!flag)
					break;
				if (tranmat[c].nnz() < mnnz) {
					mnnz = tranmat[c].nnz();
					col = c;
				}
				// make the result stable
				else if (tranmat[c].nnz() == mnnz) {
					if (col_weight(c) < col_weight(col))
						col = c;
					else if (col_weight(c) == col_weight(col) && c < col)
						col = c;
				}
			}
			if (!flag)
				continue;
			if (mnnz != SIZE_MAX) {
				pivots.emplace_back(row, col);
				dict.insert(col);
			}
		}

		std::vector<pivot_t<index_t>> result(pivots.begin(), pivots.end());
		return result;
	}

	// first choose the pivot with minimal col_weight
	// if the col_weight is negative, we do not choose it
	// only right search version, do not need the full tranpose
	template <typename T, typename index_t>
	std::vector<pivot_t<index_t>> pivots_search_right(const sparse_mat<T, index_t>& mat,
		const std::vector<index_t>& leftrows, const std::vector<index_t>& leftcols,
		const std::function<int64_t(int64_t)>& col_weight = [](int64_t i) { return i; }) {

		std::vector<pivot_t<index_t>> pivots;
		std::unordered_set<index_t> c_dict;
		c_dict.reserve((size_t)4096);

		for (auto row : leftrows) {
			index_t col = 0;
			size_t mnnz = SIZE_MAX;
			bool flag = true;

			for (auto c : mat[row].index_span()) {
				// negative weight means that we do not want to select this column
				if (col_weight(c) < 0)
					continue;
				flag = (c_dict.count(c) == 0);
				if (!flag)
					break;
				if (mnnz > 0) {
					mnnz = 0;
					col = c;
				}
				// make the result stable
				else if (mnnz == 0) {
					if (col_weight(c) < col_weight(col))
						col = c;
					else if (col_weight(c) == col_weight(col) && c < col)
						col = c;
				}
			}
			if (!flag)
				continue;
			if (mnnz != SIZE_MAX) {
				pivots.emplace_back(row, col);
				c_dict.insert(col);
			}
		}

		return pivots;
	}

	// upper solver : ordering = -1
	// lower solver : ordering = 1
	template <typename T, typename index_t>
	void triangular_solver(sparse_mat<T, index_t>& mat, 
		const std::vector<pivot_t<index_t>>& pivots,
		const field_t& F, rref_option_t opt, int ordering) {
		bool verbose = opt->verbose;
		auto printstep = opt->print_step;
		auto& pool = opt->pool;

		std::vector<std::vector<index_t>> tranmat(mat.ncol);

		// we only need to compute the transpose of the submatrix involving pivots

		for (auto& [r, c] : pivots) {
			for (auto [ind, val] : mat[r]) {
				if (val == 0)
					continue;
				tranmat[ind].push_back(r);
			}
		}

		size_t count = 0;
		size_t nthreads = pool.get_thread_count();
		for (size_t i = 0; i < pivots.size(); i++) {
			size_t index = i;
			if (ordering < 0)
				index = pivots.size() - 1 - i;
			auto [row, col] = pivots[index];
			auto& thecol = tranmat[col];
			auto start = SparseRREF::clocknow();
			if (thecol.size() > 1) {
				pool.detach_loop<index_t>(0, thecol.size(), [&](index_t j) {
					auto r = thecol[j];
					if (r == row)
						return;
					auto entry = *sparse_mat_entry(mat, r, col);
					sparse_vec_sub_mul(mat[r], mat[row], entry, F);
					},
					((thecol.size() < 20 * nthreads) ? 0 : thecol.size() / 10));
			}
			pool.wait();

			if (verbose && (i % printstep == 0 || i == pivots.size() - 1) && thecol.size() > 1) {
				count++;
				auto end = SparseRREF::clocknow();
				auto now_nnz = mat.nnz();
				std::cout << "\r-- Row: " << (i + 1) << "/" << pivots.size()
					<< "  " << "row to eliminate: " << thecol.size() - 1
					<< "  " << "nnz: " << now_nnz << "  " << "density: "
					<< (double)100 * now_nnz / (mat.nrow * mat.ncol)
					<< "%  " << "speed: " << count / SparseRREF::usedtime(start, end)
					<< " row/s" << std::flush;
				start = SparseRREF::clocknow();
				count = 0;
			}
		}
		if (opt->verbose)
			std::cout << std::endl;
	}

	template <typename T, typename index_t>
	void triangular_solver(sparse_mat<T, index_t>& mat, 
		std::vector<std::vector<pivot_t<index_t>>>& pivots,
		const field_t& F, rref_option_t opt, int ordering) {
		std::vector<pivot_t<index_t>> n_pivots;
		for (auto p : pivots)
			n_pivots.insert(n_pivots.end(), p.begin(), p.end());
		triangular_solver(mat, n_pivots, F, opt, ordering);
	}

	// dot_product
	template <typename T, typename index_t>
	sparse_vec<T, index_t> sparse_mat_dot_sparse_vec(
		const sparse_mat<T, index_t>& mat,
		const sparse_vec<T, index_t>& vec, const field_t& F) {

		sparse_vec<T, index_t> result;

		if (vec.nnz() == 0 || mat.nnz() == 0)
			return result;

		T tmp;
		for (size_t i = 0; i < mat.nrow; i++) {
			tmp = sparse_vec_dot(mat[i], vec, F);
			if (tmp != 0)
				result.push_back(i, tmp);
		}
		return result;
	}

	template <typename T, typename index_t>
	sparse_vec<T, index_t> sparse_mat_dot_dense_vec(
		const sparse_mat<T, index_t>& mat, const T* vec, const field_t& F) {

		sparse_vec<T, index_t> result;

		if (mat.nnz() == 0)
			return result;

		T tmp = 0;
		for (size_t i = 0; i < mat.nrow; i++) {
			tmp = sparse_vec_dot_dense_vec(mat[i], vec, F);
			if (tmp != 0)
				result.push_back(i, tmp);
		}
		return result;
	}

	// A = B * C
	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_mul(
		const sparse_mat<T, index_t>& B, const sparse_mat<T, index_t>& C, 
		const field_t& F, thread_pool* pool = nullptr) {

		sparse_mat<T, index_t> A(B.nrow, C.ncol);
		auto nthreads = 1;
		if (pool)
			nthreads = pool->get_thread_count();

		std::vector<T> cachedensedmat(A.ncol * nthreads);
		std::vector<SparseRREF::bit_array> nonzero_c(nthreads, A.ncol);

		auto method = [&](size_t id, size_t i) {
			auto& therow = B[i];
			if (therow.nnz() == 0)
				return;
			if (therow.nnz() == 1) {
				A[i] = C[therow(0)];
				sparse_vec_rescale(A[i], therow[0], F);
				return;
			}
			auto cache_dense_vec = cachedensedmat.data() + id * A.ncol;
			auto& nonzero_c_vec = nonzero_c[id];
			nonzero_c_vec.clear();

			T scalar = therow[0];
			ulong e_pr;
			if constexpr (std::is_same_v<T, ulong>) {
				e_pr = n_mulmod_precomp_shoup(scalar, F.mod.n);
			}
			for (auto [ind, val] : C[therow(0)]) {
				nonzero_c_vec.insert(ind);
				if constexpr (std::is_same_v<T, ulong>) {
					cache_dense_vec[ind] = n_mulmod_shoup(scalar, val, e_pr, F.mod.n);
				}
				else if constexpr (std::is_same_v<T, rat_t>) {
					cache_dense_vec[ind] = scalar * val;
				}
			}

			for (size_t j = 1; j < therow.nnz(); j++) {
				scalar = therow[j];
				if constexpr (std::is_same_v<T, ulong>) {
					e_pr = n_mulmod_precomp_shoup(scalar, F.mod.n);
				}
				for (auto [ind, val] : C[therow(j)]) {
					if (!nonzero_c_vec.test(ind)) {
						nonzero_c_vec.insert(ind);
						cache_dense_vec[ind] = 0;
					}
					if constexpr (std::is_same_v<T, ulong>) {
						cache_dense_vec[ind] = _nmod_add(cache_dense_vec[ind],
							n_mulmod_shoup(scalar, val, e_pr, F.mod.n), F.mod);
					}
					else if constexpr (std::is_same_v<T, rat_t>) {
						cache_dense_vec[ind] += scalar * val;
					}
					if (cache_dense_vec[ind] == 0)
						nonzero_c_vec.erase(ind);
				}
			}

			auto pos = nonzero_c_vec.nonzero();
			A[i].reserve(pos.size());
			A[i].resize(pos.size());
			for (size_t j = 0; j < pos.size(); j++) {
				A[i](j) = pos[j];
				A[i][j] = cache_dense_vec[pos[j]];
			}
		};

		if (pool) {
			pool->detach_loop(0, B.nrow, [&](size_t i) {
				method(SparseRREF::thread_id(), i);
				});
			pool->wait();
		}
		else {
			for (size_t i = 0; i < B.nrow; i++)
				method(0, i);
		}

		return A;
	}

	// make sure nonzero_c is cleared before calling this function
	// after this function, nonzero_c is also cleared
	template <typename T, typename index_t>
	void schur_complete(sparse_mat<T, index_t>& mat, index_t k, 
		const std::vector<pivot_t<index_t>>& pivots,
		const field_t& F, T* tmpvec, SparseRREF::bit_array& nonzero_c) {

		if (mat[k].nnz() == 0)
			return;

		for (auto [ind, val] : mat[k]) {
			nonzero_c.insert(ind);
			tmpvec[ind] = val;
		}

		ulong e_pr;
		for (auto [r, c] : pivots) {
			if (!nonzero_c.test(c))
				continue;
			T entry = tmpvec[c];
			if constexpr (std::is_same_v<T, ulong>) {
				e_pr = n_mulmod_precomp_shoup(tmpvec[c], F.mod.n);
			}
			for (auto [ind, val] : mat[r]) {
				if (!nonzero_c.test(ind)) {
					nonzero_c.insert(ind);
					tmpvec[ind] = 0;
				}
				if constexpr (std::is_same_v<T, ulong>) {
					tmpvec[ind] = _nmod_sub(tmpvec[ind],
						n_mulmod_shoup(entry, val, e_pr, F.mod.n), F.mod);
				}
				else if constexpr (std::is_same_v<T, rat_t>) {
					tmpvec[ind] -= entry * val;
				}
				else {
					tmpvec[ind] = scalar_sub(tmpvec[ind], scalar_mul(entry, val, F), F);
				}
				if (tmpvec[ind] == 0)
					nonzero_c.erase(ind);
			}
		}

		auto nnz = nonzero_c.nnz();
		if (mat[k].alloc() < nnz) {
			mat[k].reserve(nnz, false);
		}
		mat[k].resize(nnz);
		nonzero_c.nonzero_and_clear(mat[k].indices);
		for (size_t i = 0; i < nnz; i++) {
			mat[k][i] = tmpvec[mat[k](i)];
		}
	}

	// TODO: CHECK!!!
	template <typename T, typename index_t>
	void triangular_solver_2_rec(sparse_mat<T, index_t>& mat, 
		const std::vector<std::vector<index_t>>& tranmat, 
		const std::vector<pivot_t<index_t>>& pivots,
		const field_t& F, rref_option_t opt, T* cachedensedmat,
		std::vector<SparseRREF::bit_array>& nonzero_c, size_t n_split, size_t rank, size_t& process) {

		if (opt->abort)
			return;

		auto printstep = opt->print_step;
		bool verbose = opt->verbose;
		auto& pool = opt->pool;
		opt->verbose = false;
		if (pivots.size() < n_split) {
			triangular_solver(mat, pivots, F, opt, -1);
			opt->verbose = verbose;
			process += pivots.size();
			return;
		}

		std::vector<pivot_t<index_t>> sub_pivots(pivots.end() - n_split, pivots.end());
		std::vector<pivot_t<index_t>> left_pivots(pivots.begin(), pivots.end() - n_split);

		std::unordered_set<index_t> pre_leftrows;
		for (auto [r, c] : sub_pivots)
			pre_leftrows.insert(tranmat[c].begin(), tranmat[c].end());
		for (auto [r, c] : sub_pivots)
			pre_leftrows.erase(r);
		std::vector<index_t> leftrows(pre_leftrows.begin(), pre_leftrows.end());

		// for printing
		size_t now_nnz = mat.nnz();
		int bitlen_nnz = (int)std::floor(std::log(now_nnz) / std::log(10)) + 2;
		int bitlen_nrow = (int)std::floor(std::log(rank) / std::log(10)) + 1;

		auto clock_begin = SparseRREF::clocknow();
		std::atomic<size_t> cc = 0;
		pool.detach_blocks<size_t>(0, leftrows.size(), [&](const size_t s, const size_t e) {
			auto id = SparseRREF::thread_id();
			for (size_t i = s; i < e; i++) {
				schur_complete(mat, leftrows[i], sub_pivots, F, cachedensedmat + id * mat.ncol, nonzero_c[id]);
				cc++;
			}
			}, ((n_split < 20 * pool.get_thread_count()) ? 0 : leftrows.size() / 10));

		bool print_once = true;

		if (verbose) {
			size_t old_cc = cc;
			while (cc < leftrows.size() && (print_once || cc - old_cc > printstep)) {
				// stop for a while
				std::this_thread::sleep_for(std::chrono::microseconds(1000));
				now_nnz = mat.nnz();
				size_t status = (size_t)std::floor(1.0 * sub_pivots.size() * cc / leftrows.size());
				std::cout << "-- Row: " << std::setw(bitlen_nrow)
					<< process + status << "/" << rank
					<< "  nnz: " << std::setw(bitlen_nnz) << now_nnz
					<< "  density: " << std::setprecision(6) << std::setw(8)
					<< 100 * (double)now_nnz / (rank * mat.ncol) << "%"
					<< "  speed: " << std::setprecision(6) << std::setw(6)
					<< 1.0 * sub_pivots.size() * (cc - old_cc) / leftrows.size() / SparseRREF::usedtime(clock_begin, SparseRREF::clocknow())
					<< " row/s    \r" << std::flush;
				clock_begin = SparseRREF::clocknow();
				old_cc = cc;
				print_once = false;
			}
		}

		pool.wait();

		triangular_solver(mat, sub_pivots, F, opt, -1);
		opt->verbose = verbose;
		process += sub_pivots.size();

		triangular_solver_2_rec(mat, tranmat, left_pivots, F, opt, cachedensedmat, nonzero_c, n_split, rank, process);
	}

	template <typename T, typename index_t>
	void triangular_solver_2(sparse_mat<T, index_t>& mat, std::vector<pivot_t<index_t>>& pivots,
		const field_t& F, rref_option_t opt) {

		auto& pool = opt->pool;
		// prepare the tmp array
		auto nthreads = pool.get_thread_count();
		std::vector<T> cachedensedmat(mat.ncol * nthreads);
		std::vector<SparseRREF::bit_array> nonzero_c(nthreads, mat.ncol);

		if (opt->abort)
			return;

		// we only need to compute the transpose of the submatrix involving pivots
		std::vector<std::vector<index_t>> tranmat(mat.ncol);
		for (size_t i = 0; i < pivots.size(); i++) {
			auto row = pivots[i].r;
			auto indices = mat[row].indices;
			for (auto j = 0; j < mat[row].nnz(); j++) {
				tranmat[indices[j]].push_back(row);
			}
		}

		size_t process = 0;
		// TODO: better split strategy
		size_t n_split = std::max(pivots.size() / 128ULL, 1ULL << 10); 
		size_t rank = pivots.size();
		triangular_solver_2_rec(mat, tranmat, pivots, F, opt, cachedensedmat.data(), nonzero_c, n_split, rank, process);

		if (opt->verbose)
			std::cout << std::endl;
	}

	template <typename T, typename index_t>
	void triangular_solver_2(sparse_mat<T, index_t>& mat, 
		const std::vector<std::vector<pivot_t<index_t>>>& pivots,
		const field_t& F, rref_option_t opt) {

		std::vector<pivot_t<index_t>> n_pivots;
		// the first pivot is the row with only one nonzero value, so there is no need to do the elimination
		for (size_t i = 1; i < pivots.size(); i++)
			n_pivots.insert(n_pivots.end(), pivots[i].begin(), pivots[i].end());

		triangular_solver_2(mat, n_pivots, F, opt);
	}

	// TODO: TEST!!! 
	// TODO: add ordering
	// if already know the pivots, we can directly do the rref
	// if the pivots are of submatrix, it is dangerous to set clear_zero_row to true
	template <typename T, typename index_t>
	void sparse_mat_direct_rref(sparse_mat<T, index_t>& mat,
		const std::vector<std::vector<pivot_t<index_t>>>& pivots, 
		const field_t& F, rref_option_t opt, const bool clear_zero_row = true) {
		auto& pool = opt->pool;

		// first set rows not in pivots to zero
		std::vector<index_t> rowset(mat.nrow, -1);
		size_t total_rank = 0;
		for (auto p : pivots) {
			total_rank += p.size();
			for (auto [r, c] : p)
				rowset[r] = c;
		}
		if (clear_zero_row) {
			// clear the rows not in pivots
			for (size_t i = 0; i < mat.nrow; i++)
				if (rowset[i] == -1)
					mat[i].clear();
		}
		else {
			std::fill(rowset.begin(), rowset.end(), 0);
		}

		for (auto [r, c] : pivots[0]) {
			mat[r].zero();
			mat[r].push_back(c, 1);
			rowset[r] = -1;
		}

		for (size_t i = 0; i < mat.nrow; i++) {
			mat[i].compress();
		}

		std::vector<index_t> leftrows(mat.nrow, -1);
		if (opt->eliminate_one_nnz)
			eliminate_row_with_one_nnz(mat, leftrows, opt);

		leftrows.clear();

		// then do the elimination parallelly
		auto nthreads = pool.get_thread_count();
		std::vector<T> cachedensedmat(mat.ncol * nthreads);
		std::vector<SparseRREF::bit_array> nonzero_c(nthreads, mat.ncol);

		size_t rank = pivots[0].size();

		for (auto i = 1; i < pivots.size(); i++) {
			if (pivots[i].size() == 0)
				continue;

			// rescale the pivots
			pool.detach_loop(0, pivots[i].size(), [&](size_t j) {
				auto [r, c] = pivots[i][j];
				T scalar = scalar_inv(*sparse_mat_entry(mat, r, c), F);
				sparse_vec_rescale(mat[r], scalar, F);
				mat[r].reserve(mat[r].nnz());
				rowset[r] = -1;
				});
			pool.wait();

			leftrows.clear();
			for (size_t j = 0; j < mat.nrow; j++) {
				if (rowset[j] != -1 && mat[j].nnz() > 0)
					leftrows.push_back(j);
			}

			// upper solver
			// TODO: check mode
			std::atomic<size_t> cc = 0;
			size_t old_cc = cc;
			pool.detach_blocks<size_t>(0, leftrows.size(), [&](const size_t s, const size_t e) {
				auto id = SparseRREF::thread_id();
				for (size_t j = s; j < e; j++) {
					schur_complete(mat, leftrows[j], pivots[i], F, cachedensedmat.data() + id * mat.ncol, nonzero_c[id]);
					cc++;
					if (opt->abort) 
						return;
				}
				}, ((leftrows.size() < 20 * nthreads) ? 0 : leftrows.size() / 10));

			if (opt->verbose) {
				auto cn = clocknow();
				while (cc < leftrows.size()) {
					if (opt->abort) {
						pool.purge();
						return;
					}
					if (cc - old_cc > opt->print_step) {
						std::cout << "\r-- Row: "
							<< (int)std::floor(rank + (cc * 1.0 / leftrows.size()) * pivots[i].size())
							<< "/" << total_rank << "  nnz: " << mat.nnz()
							<< "  alloc: " << mat.alloc()
							<< "  speed: " << (((cc - old_cc) * 1.0 / leftrows.size()) * pivots[i].size() / usedtime(cn, clocknow()))
							<< " row/s          " << std::flush;
						old_cc = cc;
						cn = clocknow();
					}
				}
			}
			pool.wait();
			rank += pivots[i].size();
		}
		if (opt->verbose) {
			std::cout << std::endl;
		}

		if (opt->is_back_sub)
			triangular_solver_2(mat, pivots, F, opt);
	}

	template <typename T, typename index_t>
	std::vector<std::vector<pivot_t<index_t>>>
		sparse_mat_rref_forward(sparse_mat<T, index_t>& mat, const field_t& F, rref_option_t opt) {
		// first canonicalize, sort and compress the matrix

		auto& pool = opt->pool;
		auto nthreads = pool.get_thread_count();

		pool.detach_loop(0, mat.nrow, [&](auto i) { mat[i].compress(); });

		auto printstep = opt->print_step;
		bool verbose = opt->verbose;

		size_t mtx_size;
		if (pool.get_thread_count() > 16)
			mtx_size = 65536;
		else
			mtx_size = 1 << pool.get_thread_count();
		std::vector<std::mutex> mtxes(mtx_size);

		size_t now_nnz = mat.nnz();

		// store the pivots that have been used
		// -1 is not used
		std::vector<index_t> rowpivs(mat.nrow, -1);
		std::vector<std::vector<pivot_t<index_t>>> pivots;
		std::vector<pivot_t<index_t>> n_pivots;

		pool.wait();

		if (opt->abort)
			return pivots;

		// eliminate rows with only one non-zero entry
		if (opt->eliminate_one_nnz) {
			size_t count = eliminate_row_with_one_nnz_rec(mat, rowpivs, opt);
			now_nnz = mat.nnz();

			for (size_t i = 0; i < mat.nrow; i++) {
				if (opt->col_weight(rowpivs[i]) < 0)
					rowpivs[i] = -1; // mark as unused
				if (rowpivs[i] != -1)
					n_pivots.emplace_back(i, rowpivs[i]);
			}
		}
		pivots.push_back(n_pivots);

		if (opt->abort)
			return pivots;

		// use a vector to label the left columns
		std::vector<index_t> leftcols = perm_init((index_t)(mat.ncol));
		for (size_t i = 0; i < mat.nrow; i++) {
			if (rowpivs[i] != -1)
				leftcols[rowpivs[i]] = -1; // mark as used
		}
		std::erase_if(leftcols, [](index_t i) { return i == -1; });

		auto rank = pivots[0].size();
		size_t kk = rank;

		std::vector<T> cachedensedmat(mat.ncol * nthreads);
		std::vector<SparseRREF::bit_array> nonzero_c(nthreads, mat.ncol);

		std::vector<index_t> leftrows;
		leftrows.reserve(mat.nrow);
		for (size_t i = 0; i < mat.nrow; i++) {
			if (rowpivs[i] != -1 || mat[i].nnz() == 0)
				continue;
			leftrows.push_back(i);
		}

		bool only_right_search = true;

		sparse_mat<bool, index_t> tranmat;
		if (opt->method != 1) {
			only_right_search = false;
			tranmat = sparse_mat<bool, index_t>(mat.ncol, mat.nrow);
			sparse_mat_transpose_replace(tranmat, mat, &pool);

			// sort pivots by nnz, it will be faster
			std::stable_sort(leftcols.begin(), leftcols.end(),
				[&tranmat](index_t a, index_t b) {
					return tranmat[a].nnz() < tranmat[b].nnz();
				});
		}

		// for printing
		double oldpr = 0;
		int bitlen_nnz = (int)std::floor(std::log(now_nnz) / std::log(10)) + 2;
		int bitlen_ncol = (int)std::floor(std::log(mat.ncol) / std::log(10)) + 1;

		bit_array tmp_set(mat.ncol);

		while (kk < mat.ncol) {
			auto start = SparseRREF::clocknow();

			std::vector<pivot_t<index_t>> ps;

			if (opt->sort_rows) {
				std::stable_sort(leftrows.begin(), leftrows.end(), [&mat](auto a, auto b) {
					if (mat[a].nnz() < mat[b].nnz())
						return true;
					if (mat[a].nnz() == mat[b].nnz())
						return lexico_compare(mat[a].indices, mat[b].indices, mat[a].nnz()) < 0;
					return false;
					});
			}
			if (only_right_search) {
				ps = pivots_search_right(mat, leftrows, leftcols, opt->col_weight);
			}
			else {
				ps = pivots_search(mat, tranmat, rowpivs, leftrows, leftcols, opt->col_weight);
			}
			if (ps.size() == 0)
				break;

			n_pivots.clear();
			for (auto& [r, c] : ps) {
				rowpivs[r] = c;
				n_pivots.emplace_back(r, c);
			}
			pivots.push_back(n_pivots);
			rank += n_pivots.size();

			pool.detach_loop(0, n_pivots.size(), [&](size_t i) {
				auto [r, c] = n_pivots[i];
				T scalar = scalar_inv(*sparse_mat_entry(mat, r, c), F);
				sparse_vec_rescale(mat[r], scalar, F);
				mat[r].reserve(mat[r].nnz());
				});

			size_t n_leftrows = 0;
			for (size_t i = 0; i < leftrows.size(); i++) {
				auto row = leftrows[i];
				if (rowpivs[row] != -1) 
					continue;
				if (mat[row].nnz() == 0) {
					mat[row].clear();
					continue;
				}
				leftrows[n_leftrows] = row;
				n_leftrows++;
			}
			leftrows.resize(n_leftrows);
			pool.wait();

			if (opt->shrink_memory) {
				pool.detach_loop(0, leftrows.size(), [&](size_t i) {
					auto row = leftrows[i];
					if (mat[row].alloc() > 8 * mat[row].nnz()) {
						mat[row].reserve(4 * mat[row].nnz());
					}});
				pool.wait();
			}

			if (opt->abort)
				return pivots;

			if (only_right_search) {
				std::atomic<size_t> done_count = 0;
				pool.detach_blocks<size_t>(0, leftrows.size(), [&](const size_t s, const size_t e) {
					auto id = SparseRREF::thread_id();
					for (size_t i = s; i < e; i++) {
						schur_complete(mat, leftrows[i], n_pivots, F, cachedensedmat.data() + id * mat.ncol, nonzero_c[id]);
						done_count++;
						if (opt->abort)
							break;
					}
					}, (leftrows.size() < 20 * nthreads ? 0 : leftrows.size() / 10));

				// remove used cols
				size_t localcount = 0;
				tmp_set.clear();
				for (auto [r, c] : ps)
					tmp_set.insert(c);
				for (auto c : leftcols) {
					if (!tmp_set.test(c)) {
						leftcols[localcount] = c;
						localcount++;
					}
				}
				leftcols.resize(localcount);

				bool print_once = true; // print at least once

				localcount = 0;
				while (done_count < leftrows.size()) {
					if (opt->abort) {
						pool.purge();
						return pivots;
					}

					double pr = kk + (1.0 * ps.size() * done_count) / leftrows.size();
					if (verbose && (print_once || pr - oldpr > printstep)) {
						auto end = SparseRREF::clocknow();
						now_nnz = mat.nnz();
						auto now_alloc = mat.alloc();
						std::cout << "-- Col: " << std::setw(bitlen_ncol)
							<< (int)pr << "/" << mat.ncol
							<< "  rank: " << std::setw(bitlen_ncol) << rank
							<< "  nnz: " << std::setw(bitlen_nnz) << now_nnz
							<< "  alloc: " << std::setw(bitlen_nnz) << now_alloc
							<< "  density: " << std::setprecision(6) << std::setw(8)
							<< 100 * (double)now_nnz / (mat.nrow * mat.ncol) << "%"
							<< "  speed: " << std::setprecision(6) << std::setw(8) <<
							((pr - oldpr) / SparseRREF::usedtime(start, end))
							<< " col/s    \r" << std::flush;
						oldpr = pr;
						start = end;
						print_once = false;
					}

					std::this_thread::sleep_for(std::chrono::microseconds(10));
				}
				pool.wait();
			} 
			else {
				std::vector<int> flags(leftrows.size(), 0);
				pool.detach_blocks<size_t>(0, leftrows.size(), [&](const size_t s, const size_t e) {
					auto id = SparseRREF::thread_id();
					for (size_t i = s; i < e; i++) {
						schur_complete(mat, leftrows[i], n_pivots, F, cachedensedmat.data() + id * mat.ncol, nonzero_c[id]);
						flags[i] = 1;
						if (opt->abort)
							break;
					}
					}, (leftrows.size() < 20 * nthreads ? 0 : leftrows.size() / 10));

				// remove used cols
				std::atomic<size_t> localcount = 0;
				tmp_set.clear();
				for (auto [r, c] : ps)
					tmp_set.insert(c);
				for (auto c : leftcols) {
					if (!tmp_set.test(c)) {
						leftcols[localcount] = c;
						localcount++;
						tranmat[c].zero();
					}
					else
						tranmat[c].clear();
				}
				leftcols.resize(localcount);

				bool print_once = true; // print at least once

				localcount = 0;
				while (localcount < leftrows.size()) {
					// if the pool is free and too many rows left, use pool
					if (localcount * 2 < leftrows.size() && pool.get_tasks_queued() == 0) {
						std::vector<index_t> newleftrows;
						for (auto i = 0; i < leftrows.size(); i++) {
							if (flags[i])
								newleftrows.push_back(leftrows[i]);
						}

						pool.detach_loop(0, newleftrows.size(), [&](size_t i) {
							auto row = newleftrows[i];
							for (size_t j = 0; j < mat[row].nnz(); j++) {
								auto col = mat[row](j);
								std::lock_guard<std::mutex> lock(mtxes[col % mtx_size]);
								tranmat[col].push_back(row, true);
							}
							localcount++;
							}, (newleftrows.size() < 20 * nthreads ? 0 : newleftrows.size() / 10));
						pool.wait();
					}

					for (size_t i = 0; i < leftrows.size() && localcount < leftrows.size(); i++) {
						if (flags[i]) {
							auto row = leftrows[i];
							for (size_t j = 0; j < mat[row].nnz(); j++) {
								tranmat[mat[row](j)].push_back(row, true);
							}
							flags[i] = 0;
							localcount++;

							if (localcount * 2 < leftrows.size() && pool.get_tasks_queued() == 0)
								break;
						}
					}

					if (opt->abort) {
						pool.purge();
						return pivots;
					}

					double pr = kk + (1.0 * ps.size() * localcount) / leftrows.size();
					if (verbose && (print_once || pr - oldpr > printstep)) {
						auto end = SparseRREF::clocknow();
						now_nnz = mat.nnz();
						auto now_alloc = mat.alloc();
						std::cout << "-- Col: " << std::setw(bitlen_ncol)
							<< (int)pr << "/" << mat.ncol
							<< "  rank: " << std::setw(bitlen_ncol) << rank
							<< "  nnz: " << std::setw(bitlen_nnz) << now_nnz
							<< "  alloc: " << std::setw(bitlen_nnz) << now_alloc
							<< "  density: " << std::setprecision(6) << std::setw(8)
							<< 100 * (double)now_nnz / (mat.nrow * mat.ncol) << "%"
							<< "  speed: " << std::setprecision(6) << std::setw(8) <<
							((pr - oldpr) / SparseRREF::usedtime(start, end))
							<< " col/s    \r" << std::flush;
						oldpr = pr;
						start = end;
						print_once = false;
					}
				}
				pool.wait();
			}

			// if the number of new pivots is less than 1% of the total columns, 
			// it is very expansive to compute the transpose of the matrix
			// so we only search the right columns
			if (opt->method == 2 && !only_right_search && ps.size() < mat.ncol / 100) {
				only_right_search = true;
				tranmat.clear();
			}

			kk += ps.size();
		}

		if (verbose)
			std::cout << "\n** Rank: " << rank << " nnz: " << mat.nnz() << std::endl;

		return pivots;
	}

	template <typename T, typename index_t>
	std::vector<std::vector<pivot_t<index_t>>>
		sparse_mat_rref_left(sparse_mat<T, index_t>& mat, const size_t fullrank, const field_t& F, rref_option_t opt) {
		// first canonicalize, sort and compress the matrix

		auto& pool = opt->pool;
		auto nthreads = pool.get_thread_count();
		size_t p_bound = fullrank;
		std::function<int64_t(int64_t)> col_weight = [&](int64_t i) {
			if (i < p_bound)
				return opt->col_weight(i);
			else
				return (int64_t)(-1);
			};

		pool.detach_loop(0, mat.nrow, [&](auto i) { mat[i].compress(); });

		auto printstep = opt->print_step;
		bool verbose = opt->verbose;

		size_t mtx_size;
		if (pool.get_thread_count() > 16)
			mtx_size = 65536;
		else
			mtx_size = 1 << pool.get_thread_count();
		std::vector<std::mutex> mtxes(mtx_size);

		size_t now_nnz = mat.nnz();

		// store the pivots that have been used
		// -1 is not used
		std::vector<index_t> rowpivs(mat.nrow, -1);
		std::vector<std::vector<pivot_t<index_t>>> pivots;
		std::vector<pivot_t<index_t>> n_pivots;

		pool.wait();

		if (opt->abort)
			return pivots;

		if (opt->eliminate_one_nnz) {
			// eliminate rows with only one non-zero entry
			size_t count = eliminate_row_with_one_nnz_rec(mat, rowpivs, opt);
			now_nnz = mat.nnz();

			for (size_t i = 0; i < mat.nrow; i++) {
				if (opt->col_weight(rowpivs[i]) < 0)
					rowpivs[i] = -1; // mark as unused
				if (rowpivs[i] != -1)
					n_pivots.emplace_back(i, rowpivs[i]);
			}
		}
		pivots.push_back(n_pivots);

		if (opt->abort)
			return pivots;

		// use a vector to label the left columns
		std::vector<index_t> leftcols = perm_init((index_t)(mat.ncol));
		for (size_t i = 0; i < mat.nrow; i++) {
			if (rowpivs[i] != -1)
				leftcols[rowpivs[i]] = -1; // mark as used
		}
		std::erase_if(leftcols, [](index_t i) { return i == -1; });

		auto rank = pivots[0].size();
		size_t kk = rank;

		std::vector<T> cachedensedmat(mat.ncol * nthreads);
		std::vector<SparseRREF::bit_array> nonzero_c(nthreads, mat.ncol);

		std::vector<index_t> leftrows;
		leftrows.reserve(mat.nrow);
		for (size_t i = 0; i < mat.nrow; i++) {
			if (rowpivs[i] != -1 || mat[i].nnz() == 0)
				continue;
			leftrows.push_back(i);
		}

		bool only_right_search = true;

		sparse_mat<bool, index_t> tranmat;
		if (opt->method != 1) {
			only_right_search = false;
			tranmat = sparse_mat<bool, index_t>(mat.ncol, mat.nrow);
			sparse_mat_transpose_replace(tranmat, mat, &pool);

			// sort pivots by nnz, it will be faster
			std::stable_sort(leftcols.begin(), leftcols.end(),
				[&tranmat](index_t a, index_t b) {
					return tranmat[a].nnz() < tranmat[b].nnz();
				});
		}

		// for printing
		double oldpr = 0;
		int bitlen_nnz = (int)std::floor(std::log(now_nnz) / std::log(10)) + 2;
		int bitlen_ncol = (int)std::floor(std::log(mat.ncol) / std::log(10)) + 1;

		bit_array tmp_set(mat.ncol);

		while (kk < mat.ncol) {
			auto start = SparseRREF::clocknow();

			std::vector<pivot_t<index_t>> ps;

			if (opt->sort_rows) {
				std::stable_sort(leftrows.begin(), leftrows.end(), [&mat](auto a, auto b) {
					if (mat[a].nnz() < mat[b].nnz())
						return true;
					if (mat[a].nnz() == mat[b].nnz())
						return lexico_compare(mat[a].indices, mat[b].indices, mat[a].nnz()) < 0;
					return false;
					});
			}
			if (only_right_search) {
				ps = pivots_search_right(mat, leftrows, leftcols, col_weight);
			}
			else {
				ps = pivots_search(mat, tranmat, rowpivs, leftrows, leftcols, col_weight);
			}
			if (ps.size() == 0) {
				if (rank >= fullrank)
					break;
				else {
					p_bound += fullrank - rank;
					continue;
				}
			}

			n_pivots.clear();
			for (auto& [r, c] : ps) {
				rowpivs[r] = c;
				n_pivots.emplace_back(r, c);
			}
			pivots.push_back(n_pivots);
			rank += n_pivots.size();

			pool.detach_loop(0, n_pivots.size(), [&](size_t i) {
				auto [r, c] = n_pivots[i];
				T scalar = scalar_inv(*sparse_mat_entry(mat, r, c), F);
				sparse_vec_rescale(mat[r], scalar, F);
				mat[r].reserve(mat[r].nnz());
				});

			size_t n_leftrows = 0;
			for (size_t i = 0; i < leftrows.size(); i++) {
				auto row = leftrows[i];
				if (rowpivs[row] != -1)
					continue;
				if (mat[row].nnz() == 0) {
					mat[row].clear();
					continue;
				}
				leftrows[n_leftrows] = row;
				n_leftrows++;
			}
			leftrows.resize(n_leftrows);
			pool.wait();

			if (opt->shrink_memory) {
				pool.detach_loop(0, leftrows.size(), [&](size_t i) {
					auto row = leftrows[i];
					if (mat[row].alloc() > 8 * mat[row].nnz()) {
						mat[row].reserve(4 * mat[row].nnz());
					}});
					pool.wait();
			}

			if (opt->abort)
				return pivots;

			if (only_right_search) {
				std::atomic<size_t> done_count = 0;
				pool.detach_blocks<size_t>(0, leftrows.size(), [&](const size_t s, const size_t e) {
					auto id = SparseRREF::thread_id();
					for (size_t i = s; i < e; i++) {
						schur_complete(mat, leftrows[i], n_pivots, F, cachedensedmat.data() + id * mat.ncol, nonzero_c[id]);
						done_count++;
						if (opt->abort)
							break;
					}
					}, (leftrows.size() < 20 * nthreads ? 0 : leftrows.size() / 10));

				// remove used cols
				size_t localcount = 0;
				tmp_set.clear();
				for (auto [r, c] : ps)
					tmp_set.insert(c);
				for (auto c : leftcols) {
					if (!tmp_set.test(c)) {
						leftcols[localcount] = c;
						localcount++;
					}
				}
				leftcols.resize(localcount);

				bool print_once = true; // print at least once

				localcount = 0;
				while (done_count < leftrows.size()) {
					if (opt->abort) {
						pool.purge();
						return pivots;
					}

					double pr = kk + (1.0 * ps.size() * done_count) / leftrows.size();
					if (verbose && (print_once || pr - oldpr > printstep)) {
						auto end = SparseRREF::clocknow();
						now_nnz = mat.nnz();
						auto now_alloc = mat.alloc();
						std::cout << "-- Col: " << std::setw(bitlen_ncol)
							<< (int)pr << "/" << mat.ncol
							<< "  rank: " << std::setw(bitlen_ncol) << rank
							<< "  nnz: " << std::setw(bitlen_nnz) << now_nnz
							<< "  alloc: " << std::setw(bitlen_nnz) << now_alloc
							<< "  density: " << std::setprecision(6) << std::setw(8)
							<< 100 * (double)now_nnz / (mat.nrow * mat.ncol) << "%"
							<< "  speed: " << std::setprecision(6) << std::setw(8) <<
							((pr - oldpr) / SparseRREF::usedtime(start, end))
							<< " col/s    \r" << std::flush;
						oldpr = pr;
						start = end;
						print_once = false;
					}

					std::this_thread::sleep_for(std::chrono::microseconds(10));
				}
				pool.wait();
			}
			else {
				std::vector<int> flags(leftrows.size(), 0);
				pool.detach_blocks<size_t>(0, leftrows.size(), [&](const size_t s, const size_t e) {
					auto id = SparseRREF::thread_id();
					for (size_t i = s; i < e; i++) {
						schur_complete(mat, leftrows[i], n_pivots, F, cachedensedmat.data() + id * mat.ncol, nonzero_c[id]);
						flags[i] = 1;
						if (opt->abort)
							break;
					}
					}, (leftrows.size() < 20 * nthreads ? 0 : leftrows.size() / 10));

				// remove used cols
				std::atomic<size_t> localcount = 0;
				tmp_set.clear();
				for (auto [r, c] : ps)
					tmp_set.insert(c);
				for (auto c : leftcols) {
					if (!tmp_set.test(c)) {
						leftcols[localcount] = c;
						localcount++;
						tranmat[c].zero();
					}
					else
						tranmat[c].clear();
				}
				leftcols.resize(localcount);

				bool print_once = true; // print at least once

				localcount = 0;
				while (localcount < leftrows.size()) {
					// if the pool is free and too many rows left, use pool
					if (localcount * 2 < leftrows.size() && pool.get_tasks_queued() == 0) {
						std::vector<index_t> newleftrows;
						for (auto i = 0; i < leftrows.size(); i++) {
							if (flags[i])
								newleftrows.push_back(leftrows[i]);
						}

						pool.detach_loop(0, newleftrows.size(), [&](size_t i) {
							auto row = newleftrows[i];
							for (size_t j = 0; j < mat[row].nnz(); j++) {
								auto col = mat[row](j);
								std::lock_guard<std::mutex> lock(mtxes[col % mtx_size]);
								tranmat[col].push_back(row, true);
							}
							localcount++;
							}, (newleftrows.size() < 20 * nthreads ? 0 : newleftrows.size() / 10));
						pool.wait();
					}

					for (size_t i = 0; i < leftrows.size() && localcount < leftrows.size(); i++) {
						if (flags[i]) {
							auto row = leftrows[i];
							for (size_t j = 0; j < mat[row].nnz(); j++) {
								tranmat[mat[row](j)].push_back(row, true);
							}
							flags[i] = 0;
							localcount++;

							if (localcount * 2 < leftrows.size() && pool.get_tasks_queued() == 0)
								break;
						}
					}

					if (opt->abort) {
						pool.purge();
						return pivots;
					}

					double pr = kk + (1.0 * ps.size() * localcount) / leftrows.size();
					if (verbose && (print_once || pr - oldpr > printstep)) {
						auto end = SparseRREF::clocknow();
						now_nnz = mat.nnz();
						auto now_alloc = mat.alloc();
						std::cout << "-- Col: " << std::setw(bitlen_ncol)
							<< (int)pr << "/" << mat.ncol
							<< "  rank: " << std::setw(bitlen_ncol) << rank
							<< "  nnz: " << std::setw(bitlen_nnz) << now_nnz
							<< "  alloc: " << std::setw(bitlen_nnz) << now_alloc
							<< "  density: " << std::setprecision(6) << std::setw(8)
							<< 100 * (double)now_nnz / (mat.nrow * mat.ncol) << "%"
							<< "  speed: " << std::setprecision(6) << std::setw(8) <<
							((pr - oldpr) / SparseRREF::usedtime(start, end))
							<< " col/s    \r" << std::flush;
						oldpr = pr;
						start = end;
						print_once = false;
					}
				}
				pool.wait();
			}

			// if the number of new pivots is less than 1% of the total columns, 
			// it is very expansive to compute the transpose of the matrix
			// so we only search the right columns
			if (opt->method == 2 && !only_right_search && ps.size() < mat.ncol / 100) {
				only_right_search = true;
				tranmat.clear();
			}

			kk += ps.size();
		}

		if (verbose)
			std::cout << "\n** Rank: " << rank << " nnz: " << mat.nnz() << std::endl;

		return pivots;
	}

	// we assume that the column space is full rank, and want to find the most upper pivots
	template <typename T, typename index_t>
	std::vector<std::vector<pivot_t<index_t>>>
		sparse_mat_rref_up(sparse_mat<T, index_t>& mat, const size_t fullrank, const field_t& F, rref_option_t opt) {
		// first canonicalize, sort and compress the matrix

		if (fullrank == 0)
			return std::vector<std::vector<pivot_t<index_t>>>();

		auto& pool = opt->pool;
		auto nthreads = pool.get_thread_count();

		pool.detach_loop(0, mat.nrow, [&](auto i) { mat[i].compress(); });

		auto printstep = opt->print_step;
		bool verbose = opt->verbose;

		size_t p_bound = fullrank;

		size_t mtx_size;
		if (pool.get_thread_count() > 16)
			mtx_size = 65536;
		else
			mtx_size = 1 << pool.get_thread_count();
		std::vector<std::mutex> mtxes(mtx_size);

		size_t now_nnz = mat.nnz();

		// store the pivots that have been used
		// -1 is not used
		std::vector<index_t> rowpivs(mat.nrow, -1);
		std::vector<std::vector<pivot_t<index_t>>> pivots;
		std::vector<pivot_t<index_t>> n_pivots;

		pool.wait();

		if (opt->abort)
			return pivots;

		now_nnz = mat.nnz();
		pivots.push_back(n_pivots);

		if (opt->abort)
			return pivots;

		// use a vector to label the left columns
		std::vector<index_t> leftcols = perm_init((index_t)(mat.ncol));

		auto rank = pivots[0].size();
		size_t kk = rank;

		std::vector<T> cachedensedmat(mat.ncol * nthreads);
		std::vector<SparseRREF::bit_array> nonzero_c(nthreads, mat.ncol);

		std::vector<index_t> leftrows;
		leftrows.reserve(mat.nrow);
		for (size_t i = 0; i < p_bound; i++) {
			leftrows.push_back(i);
		}

		bool only_right_search = true;

		sparse_mat<bool, index_t> tranmat;
		if (opt->method != 1) {
			only_right_search = false;
			tranmat = sparse_mat<bool, index_t>(mat.ncol, mat.nrow);
			sparse_mat_transpose_part_replace(tranmat, mat, leftrows, &pool);

			// sort pivots by nnz, it will be faster
			std::stable_sort(leftcols.begin(), leftcols.end(),
				[&tranmat](index_t a, index_t b) {
					return tranmat[a].nnz() < tranmat[b].nnz();
				});
		}

		// for printing
		double oldpr = 0;
		int bitlen_nnz = (int)std::floor(std::log(now_nnz) / std::log(10)) + 2;
		int bitlen_ncol = (int)std::floor(std::log(mat.ncol) / std::log(10)) + 1;

		bit_array tmp_set(mat.ncol);
		std::vector<index_t> leftrows_ext = leftrows;

		while (kk < mat.ncol) {
			auto start = SparseRREF::clocknow();

			std::vector<pivot_t<index_t>> ps;

			if (opt->sort_rows) {
				std::stable_sort(leftrows.begin(), leftrows.end(), [&mat](auto a, auto b) {
					if (mat[a].nnz() < mat[b].nnz())
						return true;
					if (mat[a].nnz() == mat[b].nnz())
						return lexico_compare(mat[a].indices, mat[b].indices, mat[a].nnz()) < 0;
					return false;
					});
			}
			if (only_right_search) {
				ps = pivots_search_right(mat, leftrows, leftcols, opt->col_weight);
			}
			else {
				ps = pivots_search(mat, tranmat, rowpivs, leftrows, leftcols, opt->col_weight);
			}
			if (ps.size() == 0) {
				if (rank >= fullrank)
					break;
				else {
					if (p_bound == mat.nrow)
						break;
					auto np_bound = std::min(p_bound + fullrank - rank, mat.nrow);
					for (auto r = p_bound; r < np_bound; r++) {
						leftrows.push_back(r);
					}
					p_bound = np_bound;
					continue;
				}
			}

			n_pivots.clear();
			for (auto& [r, c] : ps) {
				rowpivs[r] = c;
				n_pivots.emplace_back(r, c);
			}
			pivots.push_back(n_pivots);
			rank += n_pivots.size();

			pool.detach_loop(0, n_pivots.size(), [&](size_t i) {
				auto [r, c] = n_pivots[i];
				T scalar = scalar_inv(*sparse_mat_entry(mat, r, c), F);
				sparse_vec_rescale(mat[r], scalar, F);
				mat[r].reserve(mat[r].nnz());
				});

			size_t n_leftrows = 0;
			for (size_t i = 0; i < leftrows.size(); i++) {
				auto row = leftrows[i];
				if (rowpivs[row] != -1 || mat[row].nnz() == 0)
					continue;
				leftrows[n_leftrows] = row;
				n_leftrows++;
			}
			leftrows.resize(n_leftrows);
			pool.wait();

			if (opt->shrink_memory) {
				pool.detach_loop(0, leftrows.size(), [&](size_t i) {
					auto row = leftrows[i];
					if (mat[row].alloc() > 8 * mat[row].nnz()) {
						mat[row].reserve(4 * mat[row].nnz());
					}});
					pool.wait();
			}

			if (opt->abort)
				return pivots;

			leftrows_ext = leftrows;
			for (index_t i = p_bound; i < mat.nrow; i++) {
				leftrows_ext.push_back(i);
			}

			if (only_right_search) {
				std::atomic<size_t> done_count = 0;
				pool.detach_blocks<size_t>(0, leftrows_ext.size(), [&](const size_t s, const size_t e) {
					auto id = SparseRREF::thread_id();
					for (size_t i = s; i < e; i++) {
						schur_complete(mat, leftrows_ext[i], n_pivots, F, cachedensedmat.data() + id * mat.ncol, nonzero_c[id]);
						done_count++;
						if (opt->abort)
							break;
					}
					}, (leftrows_ext.size() < 20 * nthreads ? 0 : leftrows_ext.size() / 10));

				// remove used cols
				size_t localcount = 0;
				tmp_set.clear();
				for (auto [r, c] : ps)
					tmp_set.insert(c);
				for (auto c : leftcols) {
					if (!tmp_set.test(c)) {
						leftcols[localcount] = c;
						localcount++;
					}
				}
				leftcols.resize(localcount);

				bool print_once = true; // print at least once

				localcount = 0;
				while (done_count < leftrows_ext.size()) {
					if (opt->abort) {
						pool.purge();
						return pivots;
					}

					double pr = kk + (1.0 * ps.size() * done_count) / leftrows_ext.size();
					if (verbose && (print_once || pr - oldpr > printstep)) {
						auto end = SparseRREF::clocknow();
						now_nnz = mat.nnz();
						auto now_alloc = mat.alloc();
						std::cout << "-- Col: " << std::setw(bitlen_ncol)
							<< (int)pr << "/" << mat.ncol
							<< "  rank: " << std::setw(bitlen_ncol) << rank
							<< "  nnz: " << std::setw(bitlen_nnz) << now_nnz
							<< "  alloc: " << std::setw(bitlen_nnz) << now_alloc
							<< "  density: " << std::setprecision(6) << std::setw(8)
							<< 100 * (double)now_nnz / (mat.nrow * mat.ncol) << "%"
							<< "  speed: " << std::setprecision(6) << std::setw(8) <<
							((pr - oldpr) / SparseRREF::usedtime(start, end))
							<< " col/s    \r" << std::flush;
						oldpr = pr;
						start = end;
						print_once = false;
					}

					std::this_thread::sleep_for(std::chrono::microseconds(10));
				}
				pool.wait();
			}
			else {
				std::vector<int> flags(leftrows.size(), 0);
				pool.detach_blocks<size_t>(0, leftrows_ext.size(), [&](const size_t s, const size_t e) {
					auto id = SparseRREF::thread_id();
					for (size_t i = s; i < e; i++) {
						schur_complete(mat, leftrows_ext[i], n_pivots, F, cachedensedmat.data() + id * mat.ncol, nonzero_c[id]);
						if (i < leftrows.size())
							flags[i] = 1;
						if (opt->abort)
							break;
					}
					}, (leftrows_ext.size() < 20 * nthreads ? 0 : leftrows_ext.size() / 10));

				// remove used cols
				std::atomic<size_t> localcount = 0;
				tmp_set.clear();
				for (auto [r, c] : ps)
					tmp_set.insert(c);
				for (auto c : leftcols) {
					if (!tmp_set.test(c)) {
						leftcols[localcount] = c;
						localcount++;
						tranmat[c].zero();
					}
					else
						tranmat[c].clear();
				}
				leftcols.resize(localcount);

				bool print_once = true; // print at least once

				localcount = 0;
				while (localcount < leftrows.size()) {
					// if the pool is free and too many rows left, use pool
					if (localcount * 2 < leftrows.size() && pool.get_tasks_queued() == 0) {
						std::vector<index_t> newleftrows;
						for (auto i = 0; i < leftrows.size(); i++) {
							if (flags[i])
								newleftrows.push_back(leftrows[i]);
						}

						pool.detach_loop(0, newleftrows.size(), [&](size_t i) {
							auto row = newleftrows[i];
							for (size_t j = 0; j < mat[row].nnz(); j++) {
								auto col = mat[row](j);
								std::lock_guard<std::mutex> lock(mtxes[col % mtx_size]);
								tranmat[col].push_back(row, true);
							}
							localcount++;
							}, (newleftrows.size() < 20 * nthreads ? 0 : newleftrows.size() / 10));
						pool.wait();
					}

					for (size_t i = 0; i < leftrows.size() && localcount < leftrows.size(); i++) {
						if (flags[i]) {
							auto row = leftrows[i];
							for (size_t j = 0; j < mat[row].nnz(); j++) {
								tranmat[mat[row](j)].push_back(row, true);
							}
							flags[i] = 0;
							localcount++;

							if (localcount * 2 < leftrows.size() && pool.get_tasks_queued() == 0)
								break;
						}
					}

					if (opt->abort) {
						pool.purge();
						return pivots;
					}

					double pr = kk + (1.0 * ps.size() * localcount) / leftrows.size();
					if (verbose && (print_once || pr - oldpr > printstep)) {
						auto end = SparseRREF::clocknow();
						now_nnz = mat.nnz();
						auto now_alloc = mat.alloc();
						std::cout << "-- Col: " << std::setw(bitlen_ncol)
							<< (int)pr << "/" << mat.ncol
							<< "  rank: " << std::setw(bitlen_ncol) << rank
							<< "  nnz: " << std::setw(bitlen_nnz) << now_nnz
							<< "  alloc: " << std::setw(bitlen_nnz) << now_alloc
							<< "  density: " << std::setprecision(6) << std::setw(8)
							<< 100 * (double)now_nnz / (mat.nrow * mat.ncol) << "%"
							<< "  speed: " << std::setprecision(6) << std::setw(8) <<
							((pr - oldpr) / SparseRREF::usedtime(start, end))
							<< " col/s    \r" << std::flush;
						oldpr = pr;
						start = end;
						print_once = false;
					}
				}
				pool.wait();
			}

			// if the number of new pivots is less than 1% of the total columns, 
			// it is very expansive to compute the transpose of the matrix
			// so we only search the right columns
			if (opt->method == 2 && !only_right_search && ps.size() < mat.ncol / 100) {
				only_right_search = true;
				tranmat.clear();
			}

			kk += ps.size();
		}

		if (verbose)
			std::cout << "\n** Rank: " << rank << " nnz: " << mat.nnz() << std::endl;

		return pivots;
	}

	template <typename T, typename index_t>
	std::vector<std::vector<pivot_t<index_t>>> 
		sparse_mat_rref(sparse_mat<T, index_t>& mat, const field_t& F, rref_option_t opt) {

		std::vector<std::vector<pivot_t<index_t>>> pivots;
		pivots = sparse_mat_rref_forward(mat, F, opt);

		if (opt->shrink_memory) {
			opt->pool.detach_loop(0, mat.nrow, [&](auto i) {
					mat[i].reserve(mat[i].nnz());
				});
			opt->pool.wait();
		}

		if (opt->abort) 
			return pivots;

		if (opt->is_back_sub) {
			if (opt->verbose)
				std::cout << "\n>> Reverse solving: " << std::endl;
			// triangular_solver(mat, pivots, F, opt, -1);
			triangular_solver_2(mat, pivots, F, opt);
		}
		return pivots;
	}

	// TODO: check!!! 
	// checkrank only used for sparse_mat_inverse

	// TODO: if the height of initial matrix is too large, the result may be wrong,
	// since it need to more primes to reconstruct the matrix. 
	// The correct condition: H(d*E)*H(mat)*ncol < product of primes
	// where H is the height of a matrix, E is the reconstracted rref matrix
	// d the denominartor of E such that d*E is a integer matrix
	template <typename index_t>
	std::vector<std::vector<pivot_t<index_t>>> sparse_mat_rref_reconstruct(
		sparse_mat<rat_t, index_t>& mat, rref_option_t opt, const bool checkrank = false) {
		std::vector<std::vector<pivot_t<index_t>>> pivots;

		auto& pool = opt->pool;
		auto nthreads = pool.get_thread_count();

		pool.detach_loop(0, mat.nrow, [&](auto i) { mat[i].compress(); });
		pool.wait();

		ulong prime = n_nextprime(1ULL << 60, 0);
		field_t F(FIELD_Fp, prime);

		sparse_mat<ulong, index_t> matul(mat.nrow, mat.ncol);
		pool.detach_loop(0, mat.nrow, [&](auto i) {
			matul[i] = mat[i] % F.mod;
			});
		pool.wait();

		ulong mat_height_bits = mat.height_bits() + 64 - std::countl_zero(mat.ncol);

		pivots = sparse_mat_rref_forward(matul, F, opt);

		if (opt->abort) 
			return pivots;

		if (checkrank) {
			size_t rank = 0;
			for (auto& p : pivots) 
				rank += p.size();
			if (rank != mat.nrow) 
				return pivots;
		}

		if (opt->is_back_sub)
			triangular_solver_2(matul, pivots, F, opt);

		int_t mod = prime;

		bool isok = true;
		sparse_mat<rat_t, index_t> matq(mat.nrow, mat.ncol);

		std::vector<index_t> leftrows;

		for (auto i = 0; i < mat.nrow; i++) {
			size_t nnz = matul[i].nnz();
			if (nnz == 0)
				continue;
			leftrows.push_back(i);
			matq[i].reserve(nnz);
			matq[i].resize(nnz);
			for (size_t j = 0; j < nnz; j++) {
				matq[i](j) = matul[i](j);
				int_t mod1 = matul[i][j];
				if (isok) 
					isok = rational_reconstruct(matq[i][j], mod1, mod);
			}
		}

		sparse_mat<int_t, index_t> matz(mat.nrow, mat.ncol);
		if (!isok || mod.bits() < mat_height_bits) {
			for (auto i = 0; i < mat.nrow; i++)
				matz[i] = matul[i];
		}

		auto verbose = opt->verbose;

		if (verbose) {
			std::cout << std::endl;
		}

		if (opt->abort)
			return pivots;

		ulong old_height = matq.height_bits();

		// set rows not in pivots to zero
		if (!isok || mod.bits() < mat_height_bits) {
			std::vector<index_t> rowset(mat.nrow, -1);
			for (auto p : pivots)
				for (auto [r, c] : p)
					rowset[r] = c;
			for (size_t i = 0; i < mat.nrow; i++)
				if (rowset[i] == -1) {
					mat[i].clear();
				}
		}

		while (!isok || mod.bits() < mat_height_bits) {
			isok = true;
			prime = n_nextprime(prime, 0);
			if (verbose)
				std::cout << ">> Reconstruct failed, try next prime: " << prime << std::endl;
			int_t mod1 = mod * prime;
			F = field_t(FIELD_Fp, prime);
			pool.detach_loop(0, mat.nrow, [&](auto i) {
				matul[i] = mat[i] % F.mod;
				});
			pool.wait();
			sparse_mat_direct_rref(matul, pivots, F, opt);
			if (opt->is_back_sub) {
				opt->verbose = false;
				triangular_solver_2(matul, pivots, F, opt);
			}
			std::vector<int> flags(nthreads, 1);

			pool.detach_loop<size_t>(0, leftrows.size(), [&](size_t i) {
				size_t row = leftrows[i];
				auto id = SparseRREF::thread_id();
				for (size_t j = 0; j < matul[row].nnz(); j++) {
					matz[row][j] = CRT(matz[row][j], mod, matul[row][j], prime);
					if (flags[id])
						flags[id] = rational_reconstruct(matq[row][j], matz[row][j], mod1);
				}
				});

			pool.wait();
			for (auto f : flags)
				isok = isok && f;

			mod = mod1;

			if (matq.height_bits() > old_height) {
				isok = false;
				old_height = matq.height_bits();
			}
		}
		opt->verbose = verbose;

		if (opt->verbose) {
			std::cout << "** Reconstruct success! Using mod ~ "
				<< "2^" << mod.bits() << ".                " << std::endl;
		}

		mat = matq;

		return pivots;
	}

	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_rref_kernel(const sparse_mat<T, index_t>& M,
		const std::vector<pivot_t<index_t>>& pivots, const field_t& F, rref_option_t opt) {

		auto& pool = opt->pool;

		sparse_mat<T, index_t> K;
		auto rank = pivots.size();
		if (rank == M.ncol)
			return K;

		if (rank == 0) {
			K.init(M.ncol, M.ncol);
			for (size_t i = 0; i < M.ncol; i++)
				K[i].push_back(i, (T)1);
			return K;
		}
		T m1 = scalar_neg((T)1, F);

		sparse_mat<T, index_t> rows(rank, M.ncol);
		sparse_mat<T, index_t> trows(M.ncol, rank);
		for (size_t i = 0; i < rank; i++) {
			rows[i] = M[pivots[i].r];
		}
		sparse_mat_transpose_replace(trows, rows);

		std::vector<index_t> colpivs(M.ncol, -1);
		std::vector<index_t> nonpivs;
		for (size_t i = 0; i < rank; i++)
			colpivs[pivots[i].c] = pivots[i].r;

		for (auto i = 0; i < M.ncol; i++)
			if (colpivs[i] == -1)
				nonpivs.push_back(i);

		K.init(M.ncol - rank, M.ncol);
		pool.detach_loop<size_t>(0, nonpivs.size(), [&](size_t i) {
			auto& thecol = trows[nonpivs[i]];
			K[i].reserve(thecol.nnz() + 1);
			for (size_t j = 0; j < thecol.nnz(); j++) {
				K[i].push_back(pivots[thecol(j)].c, thecol[j]);
			}
			K[i].push_back(nonpivs[i], m1);
			K[i].sort_indices();
			});
		pool.wait();

		return K;
	}

	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_rref_kernel(const sparse_mat<T, index_t>& M,
		const std::vector<std::vector<pivot_t<index_t>>>& pivots, 
		const field_t& F, rref_option_t opt) {
		std::vector<pivot_t<index_t>> n_pivots;
		for (auto& p : pivots)
			n_pivots.insert(n_pivots.end(), p.begin(), p.end());
		return sparse_mat_rref_kernel(M, n_pivots, F, opt);
	}

	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_inverse(const sparse_mat<T, index_t>& M, 
		const field_t& F, rref_option_t opt) {
		if (M.nrow != M.ncol) {
			std::cerr << "Error: sparse_mat_inverse: matrix is not square" << std::endl;
			return sparse_mat<T, index_t>();
		}

		auto& pool = opt->pool;

		// define the Augmented matrix
		auto M1 = M;
		M1.compress();
		for (size_t i = 0; i < M1.nrow; i++) {
			if (M1[i].nnz() == 0) {
				std::cerr << "Error: sparse_mat_inverse: matrix is not invertible" << std::endl;
				return sparse_mat<T, index_t>();
			}
			M1[i].push_back(i + M.ncol, (T)1);
		}
		M1.ncol *= 2;

		// backup the option
		auto old_col_weight = opt->col_weight;
		std::function<int64_t(int64_t)> new_col_weight = [&](int64_t i) {
			return ((i < M.ncol) ? old_col_weight(i) : -1);
			};
		opt->col_weight = new_col_weight;
		bool is_back_sub = opt->is_back_sub;
		opt->is_back_sub = true;

		std::vector<std::vector<pivot_t<index_t>>> pivots;
		if (F.ring == RING::FIELD_Fp)
			pivots = sparse_mat_rref(M1, F, opt);
		else if (F.ring == RING::FIELD_QQ)
			pivots = sparse_mat_rref_reconstruct(M1, opt, true);
		else {
			std::cerr << "Error: sparse_mat_inverse: field not supported" << std::endl;
			return sparse_mat<T, index_t>();
		}

		std::vector<pivot_t<index_t>> flatten_pivots;
		size_t rank = 0;
		for (auto& p : pivots) {
			rank += p.size();
			flatten_pivots.insert(flatten_pivots.end(), p.begin(), p.end());
		}
			
		if (rank != M.nrow) {
			std::cerr << "Error: sparse_mat_inverse: matrix is not invertible" << std::endl;
			return sparse_mat<T, index_t>();
		}

		auto perm = perm_init(M.nrow);
		std::sort(perm.begin(), perm.end(),
			[&](size_t a, size_t b) {
				return flatten_pivots[a].c < flatten_pivots[b].c;
			});
		for (size_t i = 0; i < M.nrow; i++) {
			perm[i] = flatten_pivots[perm[i]].r;
		}

		permute(perm, M1.rows);

		for (size_t i = 0; i < M1.nrow; i++) {
			// the first ncol columns is the identity matrix,
			// we need to remove it
			for (size_t j = 0; j < M1[i].nnz() - 1; j++) {
				M1[i][j] = M1[i][j + 1];
				M1[i](j) = M1[i](j + 1) - M.ncol;
			}
			M1[i].resize(M1[i].nnz() - 1);
		}
		M1.ncol = M.ncol;

		// restore the option
		opt->col_weight = old_col_weight;
		opt->is_back_sub = is_back_sub;

		return M1;
	}

	// IO
	template <typename ScalarType, typename index_t, typename T>
	sparse_mat<ScalarType, index_t> sparse_mat_read(T& st, const field_t& F) {
		if (!st.is_open()) {
			std::cerr << "Error: sparse_mat_read: file not open." << std::endl;
			return sparse_mat<ScalarType, index_t>();
		}

		std::string line;
		std::vector<size_t> dims;
		sparse_mat<ScalarType, index_t> mat;

		while (std::getline(st, line)) {
			if (line.empty() || line[0] == '%')
				continue;

			size_t start = 0;
			size_t end = line.find(' ');
			while (end != std::string::npos) {
				if (start != end) {
					dims.push_back(string_to_ull(line.substr(start, end - start)));
				}
				start = end + 1;
				end = line.find(' ', start);
			}
			if (start < line.size()) {
				// size_t nnz = string_to_ull(line.substr(start));
				if (dims.size() != 2) {
					std::cerr << "Error: sparse_mat_read: wrong format in the matrix file" << std::endl;
					return sparse_mat<ScalarType, index_t>();
				}
				mat = sparse_mat<ScalarType, index_t>(dims[0], dims[1]);
			}
			break;
		}

		while (std::getline(st, line)) {
			if (line.empty() || line[0] == '%')
				continue;

			bool is_end = false;
			size_t rowcol[2];
			size_t* rowcolptr = rowcol;
			size_t start = 0;
			size_t end = line.find(' ');
			size_t count = 0;

			while (end != std::string::npos && count < 2) {
				if (start != end) {
					auto val = string_to_ull(line.substr(start, end - start));
					if (val == 0) {
						is_end = true;
						break;
					}
					*rowcolptr = val - 1;
					rowcolptr++;
					count++;
				}
				start = end + 1;
				end = line.find(' ', start);
			}

			if (is_end)
				break;

			if (count != 2) {
				std::cerr << "Error: sparse_mat_read: wrong format in the matrix file" << std::endl;
				return sparse_mat<ScalarType, index_t>();
			}

			ScalarType val;
			if constexpr (std::is_same_v<ScalarType, ulong>) {
				rat_t raw_val(line.substr(start));
				val = raw_val % F.mod;
			}
			else if constexpr (std::is_same_v<ScalarType, rat_t>) {
				val = rat_t(line.substr(start));
			}

			mat[rowcol[0]].push_back(rowcol[1], val);
		}

		return mat;
	}

	// SparseArray[Automatic,dims,imp_val = 0,{1,{rowptr,colindex},vals}]
	// TODO: more check!!!
	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_read_wxf(const WXF_PARSER::ExprTree& tree, const field_t& F) {
		auto& root = tree.root;

		// SparseArray
		std::string tmp_str(tree[root].str);

		if (tmp_str != "SparseArray") { 
			std::cerr << "Error: sparse_mat_read: ";
			std::cerr << "not a SparseArray with rational / integer entries" << std::endl;
			return sparse_mat<T, index_t>();
		}

		// dims
		std::vector<index_t> dims(tree[root[1]].i_arr, tree[root[1]].i_arr + tree[root[1]].dim(0));

		if (tree[root[2]].i != 0) {
			std::cerr << "Error: sparse_mat_read: the implicit value is not 0" << std::endl;
			return sparse_mat<T, index_t>();
		}

		// {1,{rowptr,colindex},vals}
		auto& last_node = root[3];

		// last_node[0] should be 1, what's the meaning of this?

		// last_node[1] is {rowptr,colindex}
		// last_node[1][0] is rowptr, last_node[1][1] is colindex
		std::vector<index_t> rowptr(tree[last_node[1][0]].i_arr, tree[last_node[1][0]].i_arr
			+ tree[last_node[1][0]].dim(0));
		std::vector<index_t> colindex(tree[last_node[1][1]].i_arr, tree[last_node[1][1]].i_arr
			+ tree[last_node[1][1]].dim(0));
	
		auto toInteger = [](const WXF_PARSER::TOKEN& node) {
			switch (node.type) {
			case WXF_PARSER::i8:
			case WXF_PARSER::i16:
			case WXF_PARSER::i32:
			case WXF_PARSER::i64:
				return Flint::int_t(node.i);
			case WXF_PARSER::bigint:
				return Flint::int_t(node.str);
			default:
				std::cerr << "not a integer" << std::endl;
				return Flint::int_t(0);
			}
			};

		// last_node[2] is vals
		std::vector<T> vals;
		if (tree[last_node[2]].type == WXF_PARSER::array ||
			tree[last_node[2]].type == WXF_PARSER::narray) {

			auto ptr = tree[last_node[2]].i_arr;
			auto nnz = tree[last_node[2]].dim(0);

			vals.resize(nnz);
			if constexpr (std::is_same_v<T, rat_t>) {
				for (size_t i = 0; i < nnz; i++) {
					vals[i] = ptr[i];
				}
			}
			else if constexpr (std::is_same_v<T, ulong>) {
				for (size_t i = 0; i < nnz; i++) {
					vals[i] = int_t(ptr[i]) % F.mod;
				}
			}
		}
		else {
			auto nnz = last_node[2].size;
			vals.resize(nnz);

			for (size_t i = 0; i < nnz; i++) {
				T val;
				auto& val_node = last_node[2][i];
				auto& token = tree[val_node];

				switch (tree[val_node].type) {
				case WXF_PARSER::i8:
				case WXF_PARSER::i16:
				case WXF_PARSER::i32:
				case WXF_PARSER::i64: 
					if constexpr (std::is_same_v<T, rat_t>) {
						val = token.i;
					}
					else if constexpr (std::is_same_v<T, ulong>) {
						val = int_t(token.i) % F.mod;
					}
					break;
				case WXF_PARSER::bigint:
					if constexpr (std::is_same_v<T, rat_t>) {
						val = toInteger(token);
					}
					else if constexpr (std::is_same_v<T, ulong>) {
						val = int_t(token.str) % F.mod;
					}
					break;
				case WXF_PARSER::symbol:
					tmp_str = token.str;
					if (tmp_str == "Rational") {
						int_t n_1 = toInteger(tree[val_node[0]]);
						int_t d_1 = toInteger(tree[val_node[1]]);
						if constexpr (std::is_same_v<T, rat_t>) {
							val = rat_t(std::move(n_1), std::move(d_1), true);
						}
						else if constexpr (std::is_same_v<T, ulong>) {
							val = rat_t(std::move(n_1), std::move(d_1), true) % F.mod;
						}
					}
					else {
						std::cerr << "Error: sparse_mat_read: ";
						std::cerr << "not a SparseArray with rational / integer entries" << std::endl;
						return sparse_mat<T, index_t>();
					}
					break;
				default:
					std::cerr << "Error: sparse_mat_read: ";
					std::cerr << "not a SparseArray with rational / integer entries" << std::endl;
					return sparse_mat<T, index_t>();
					break;
				}
				vals[i] = val;
			}
		}

		sparse_mat<T, index_t> mat(dims[0], dims[1]);
		for (size_t i = 0; i < rowptr.size() - 1; i++) {
			for (size_t j = rowptr[i]; j < rowptr[i + 1]; j++) {
				// mathematica is 1-indexed
				mat[i].push_back(colindex[j] - 1, vals[j]);
			}
		}

		return mat;
	}

	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_read_wxf(const std::filesystem::path file, const field_t& F) {
		return sparse_mat_read_wxf<T, index_t>(WXF_PARSER::MakeExprTree(file), F);
	}

	// SparseArray[Automatic,dims,imp_val = 0,{1,{rowptr,colindex},vals}]
	// TODO: more check!!!
	template <typename T, typename index_t>
	std::vector<uint8_t> sparse_mat_write_wxf(const sparse_mat<T, index_t>& mat, bool include_head = true) {
		std::vector<uint8_t> buffer;
		std::vector<uint8_t> short_buffer;
		short_buffer.reserve(16);
		uint64_t mat_nnz = mat.nnz();
		buffer.reserve(mat_nnz * 16); // reserve some space for the data
		std::string tmp_str;
		tmp_str.reserve(256);

		using WXF_PARSER::serialize_binary;

		auto toVarint = [&](uint64_t value) {
			short_buffer.clear();
			auto tmp_val = value;
			while (tmp_val > 0) {
				uint8_t byte = tmp_val & 127;
				tmp_val >>= 7;
				if (tmp_val > 0) byte |= 128; // set the continuation bit
				short_buffer.push_back(byte);
			}
			};

		auto push_symbol = [&](const std::string& str) {
			buffer.push_back(WXF_PARSER::symbol); buffer.push_back(str.size());
			buffer.insert(buffer.end(), str.begin(), str.end());
			};
		auto push_varint = [&](uint64_t size) {
			toVarint(size);
			buffer.insert(buffer.end(), short_buffer.begin(), short_buffer.end());
			};
		auto push_function = [&](const std::string& symbol, uint64_t size) {
			buffer.push_back(WXF_PARSER::func); 
			push_varint(size);
			push_symbol(symbol);
			};
		auto push_integer = [&](const int_t& num) {
			if (num.fits_si()) {
				// int64_t int
				buffer.push_back(WXF_PARSER::i64);
				serialize_binary(buffer, num.to_si());
			}
			else {
				// big integer
				buffer.push_back(WXF_PARSER::bigint);
				std::string str = num.get_str();
				push_varint(str.size());
				auto old_size = buffer.size();
				buffer.resize(old_size + str.size());
				std::memcpy(buffer.data() + old_size, str.data(), str.size());
			}
			};

		// header
		if (include_head) {
			buffer.push_back(56); buffer.push_back(58);
		}
		// function, 4, "SparseArray"
		push_function("SparseArray", 4);
		// symbol, 9, "Automatic"
		push_symbol("Automatic");

		// dims
		buffer.push_back(WXF_PARSER::array); 
		int bit_len = std::countr_zero(sizeof(size_t) / sizeof(char));
		buffer.push_back(bit_len); // 3 for int64_t, 2 for int32_t ...
		buffer.push_back(1); // rank 1
		buffer.push_back(2); // 2 for 2D array
		serialize_binary(buffer, mat.nrow);
		serialize_binary(buffer, mat.ncol);
		// implicit value, 0
		buffer.push_back(WXF_PARSER::i8); buffer.push_back(0);
		// function, List, 3
		push_function("List", 3);
		// i8, 1
		buffer.push_back(WXF_PARSER::i8); buffer.push_back(1);
		// function, List, 2
		push_function("List", 2);

		// rowptr
		buffer.push_back(WXF_PARSER::array);
		bit_len = std::countr_zero(sizeof(size_t) / sizeof(char));
		buffer.push_back(bit_len); // 3 for int64_t, 2 for int32_t ...
		buffer.push_back(1); // rank 1
		push_varint(mat.nrow + 1);
		uint64_t tmp_rowptr = 0;
		serialize_binary(buffer, tmp_rowptr);
		for (size_t i = 0; i < mat.nrow; i++) {
			tmp_rowptr += mat[i].nnz();
			serialize_binary(buffer, tmp_rowptr);
		}

		// colindex
		buffer.push_back(WXF_PARSER::array);
		bit_len = std::countr_zero(sizeof(index_t) / sizeof(char));
		buffer.push_back(bit_len); // 3 for int64_t, 2 for int32_t ...
		buffer.push_back(2); // rank 2
		// {nnz,1}
		push_varint(mat_nnz);
		buffer.push_back(1);
		// data of colindex
		for (size_t i = 0; i < mat.nrow; i++) {
			for (size_t j = 0; j < mat[i].nnz(); j++) {
				serialize_binary(buffer, mat[i](j) + 1); // mathematica is 1-indexed
			}
		}

		// vals
		if constexpr (std::is_same_v<T, ulong>) {
			buffer.push_back(WXF_PARSER::array);
			bit_len = std::countr_zero(sizeof(ulong) / sizeof(char));
			buffer.push_back(bit_len); // 3 for int64_t, 2 for int32_t ...
			// {1, nnz}
			buffer.push_back(1); 
			push_varint(mat_nnz);
			// data of vals
			for (size_t i = 0; i < mat.nrow; i++) {
				auto old_size = buffer.size();
				buffer.resize(old_size + mat[i].nnz() * sizeof(ulong));
				std::memcpy(buffer.data() + old_size, mat[i].entries, mat[i].nnz() * sizeof(ulong));
			}
		}
		else if constexpr (std::is_same_v<T, rat_t>) {
			// use List to store the vals
			// function, List, nnz
			push_function("List", mat_nnz);

			// data of vals
			for (size_t i = 0; i < mat.nrow; i++) {
				for (size_t j = 0; j < mat[i].nnz(); j++) {
					rat_t rat_val = mat[i][j];
					int_t num = rat_val.num();
					int_t den = rat_val.den();
					if (den == 1) {
						push_integer(num);
					}
					else {
						// function, Rational, 2
						push_function("Rational", 2);
						push_integer(num);
						push_integer(den);
					}
				}
			}
		}

		buffer.shrink_to_fit();
		return buffer;
	}

	template <typename T, typename S, typename index_t>
	void sparse_mat_write(sparse_mat<T, index_t>& mat, S& st, enum SPARSE_FILE_TYPE type) {
		switch (type) {
		case SPARSE_FILE_TYPE_PLAIN:
			st << mat.nrow << ' ' << mat.ncol << ' ' << mat.nnz() << '\n';
			break;
		case SPARSE_FILE_TYPE_MTX:
			if constexpr (std::is_same_v<T, ulong>) {
				st << "%%MatrixMarket matrix coordinate integer general\n";
			}
			st << mat.nrow << ' ' << mat.ncol << ' ' << mat.nnz() << '\n';
			break;
		case SPARSE_FILE_TYPE_SMS: {
			char type_char =
				std::is_same_v<T, rat_t> ? 'Q' :
				(std::is_same_v<T, ulong> || std::is_same_v<T, int_t>) ? 'M' : '\0';
			if (type_char == '\0') {
				return;
			}
			st << mat.nrow << ' ' << mat.ncol << ' ' << type_char << '\n';
			break;
		}
		default:
			return;
		}

		char num_buf[32];
		std::string line_buf;
		line_buf.reserve(mat.nnz() * 32);

		for (size_t i = 0; i < mat.nrow; ++i) {
			for (auto [ind, val] : mat[i]) {
				if (val == 0) continue;

				auto [ptr1, ec1] = std::to_chars(num_buf, num_buf + sizeof(num_buf), i + 1);
				line_buf.append(num_buf, ptr1);
				line_buf.push_back(' ');

				auto [ptr2, ec2] = std::to_chars(num_buf, num_buf + sizeof(num_buf), ind + 1);
				line_buf.append(num_buf, ptr2);
				line_buf.push_back(' ');

				if constexpr (std::is_integral_v<T>) {
					auto [ptr3, ec3] = std::to_chars(num_buf, num_buf + sizeof(num_buf), val);
					line_buf.append(num_buf, ptr3);
				}
				else if constexpr (std::is_same_v<T, rat_t> || std::is_same_v<T, int_t>) {
					line_buf += val.get_str();
				}

				line_buf.push_back('\n');
			}
		}

		st.write(line_buf.data(), line_buf.size());

		if (type == SPARSE_FILE_TYPE_SMS) {
			st << "0 0 0\n";
		}
	}

	
	static std::pair<char*, char*> snmod_mat_to_binary(sparse_mat<ulong>& mat) {
		auto ratio_i = sizeof(slong) / sizeof(char);
		auto ratio_e = sizeof(ulong) / sizeof(char);
		auto nnz = mat.nnz();
		auto len = 3 * ratio_e + mat.nrow * ratio_e + nnz * (ratio_i + ratio_e);
		char* buffer = s_malloc<char>(len);
		char* ptr = buffer;
		ulong some_n[3] = { mat.nrow, mat.ncol, nnz };
		std::memcpy(ptr, some_n, 3 * sizeof(ulong)); ptr += 3 * ratio_e;
		for (size_t i = 0; i < mat.nrow; i++) 
			ptr = snmod_vec_to_binary(mat[i], ptr).second;
		return std::make_pair(buffer, ptr);
	}

	
	sparse_mat<ulong> snmod_mat_from_binary(char* buffer) {
		auto ratio_i = sizeof(slong) / sizeof(char);
		auto ratio_e = sizeof(ulong) / sizeof(char);
		char* ptr = buffer;
		ulong some_n[3]; // nrow, ncol, nnz
		std::memcpy(some_n, ptr, 3 * sizeof(ulong)); ptr += 3 * ratio_e;
		sparse_mat<ulong> mat(some_n[0], some_n[1]);
		for (size_t i = 0; i < mat.nrow; i++)
			ptr = snmod_vec_from_binary(mat[i], ptr);

		return mat;
	}

} // namespace SparseRREF

#endif