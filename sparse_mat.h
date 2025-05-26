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
	template <typename T, typename index_type>
	inline T* sparse_mat_entry(sparse_mat<T, index_type>& mat, size_t r, index_type c, bool isbinary = true) {
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

		constexpr size_t mtx_size = 128;
		std::mutex mtxes[mtx_size];
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
		int bitlen_nnz = (int)std::floor(std::log(oldnnz) / std::log(10)) + 3;
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
		} while (localcounter > 0 && depth < max_depth && localcounter > 10);
		return count;
	}

	// first choose the pivot with minimal col_weight
	// if the col_weight is negative, we do not choose it
	template <typename T, typename index_t>
	std::vector<std::pair<index_t, index_t>> findmanypivots(
		const sparse_mat<T, index_t>& mat, const sparse_mat<bool, index_t>& tranmat,
		std::vector<index_t>& rowpivs, std::vector<index_t>& leftcols,
		const std::function<int64_t(int64_t)>& col_weight = [](int64_t i) { return i; }) {

		auto nrow = mat.nrow;
		auto ncol = mat.ncol;

		std::list<std::pair<index_t, index_t>> pivots;
		std::unordered_set<index_t> dict;
		dict.reserve((size_t)4096);

		// rightlook first
		for (auto dir : leftcols) {
			if (tranmat[dir].nnz() == 0)
				continue;

			index_t rdiv;
			size_t mnnz = SIZE_MAX;
			bool flag = true;

			for (auto col : tranmat[dir].index_span()) {
				flag = (dict.count(col) == 0);
				if (!flag)
					break;
				if (rowpivs[col] != -1)
					continue;
				size_t newnnz = mat[col].nnz();
				if (newnnz < mnnz) {
					// negative weight means that we do not want to select this column
					if (col_weight(col) < 0)
						continue;
					rdiv = col;
					mnnz = newnnz;
				}
				// make the result stable
				else if (newnnz == mnnz) {
					if (col_weight(col) < 0)
						continue;
					if (col_weight(col) < col_weight(rdiv))
						rdiv = col;
					else if (col_weight(col) == col_weight(rdiv) && col < rdiv)
						rdiv = col;
				}
			}
			if (!flag)
				continue;
			if (mnnz != SIZE_MAX) {
				pivots.push_back(std::make_pair(rdiv, dir));
				dict.insert(rdiv);
			}
		}

		// leftlook then
		dict.clear();
		// make a table to help to look for dir pointers
		std::vector<index_t> colptrs(ncol, -1);
		for (auto col : leftcols)
			colptrs[col] = col;

		for (auto p : pivots)
			dict.insert(p.second);

		for (size_t i = 0; i < nrow; i++) {
			auto row = i;
			// auto rdir = nrdir - i - 1; // reverse ordering
			if (rowpivs[row] != -1)
				continue;

			index_t dir = 0;
			size_t mnnz = SIZE_MAX;
			bool flag = true;

			for (auto col : mat[row].index_span()) {
				if (colptrs[col] == -1)
					continue;
				flag = (dict.count(col) == 0);
				if (!flag)
					break;
				if (tranmat[col].nnz() < mnnz) {
					// negative weight means that we do not want to select this column
					if (col_weight(col) < 0)
						continue;
					mnnz = tranmat[col].nnz();
					dir = col;
				}
				// make the result stable
				else if (tranmat[col].nnz() == mnnz) {
					if (col_weight(col) < 0)
						continue;
					if (col_weight(col) < col_weight(dir))
						dir = col;
					else if (col_weight(col) == col_weight(dir) && col < dir)
						dir = col;
				}
			}
			if (!flag)
				continue;
			if (mnnz != SIZE_MAX) {
				pivots.push_front(std::make_pair(row, dir));
				dict.insert(dir);
			}
		}

		std::vector<std::pair<index_t, index_t>> result(pivots.begin(), pivots.end());
		return result;
	}

	// upper solver : ordering = -1
	// lower solver : ordering = 1
	template <typename T, typename index_t>
	void triangular_solver(sparse_mat<T, index_t>& mat, 
		std::vector<std::pair<index_t, index_t>>& pivots,
		const field_t F, rref_option_t opt, int ordering) {
		bool verbose = opt->verbose;
		auto printstep = opt->print_step;
		auto& pool = opt->pool;

		std::vector<std::vector<index_t>> tranmat(mat.ncol);

		// we only need to compute the transpose of the submatrix involving pivots

		for (auto p : pivots) {
			for (auto [ind, val] : mat[p.first]) {
				if (val == 0)
					continue;
				tranmat[ind].push_back(p.first);
			}
		}

		size_t count = 0;
		size_t nthreads = pool.get_thread_count();
		for (size_t i = 0; i < pivots.size(); i++) {
			size_t index = i;
			if (ordering < 0)
				index = pivots.size() - 1 - i;
			auto pp = pivots[index];
			auto& thecol = tranmat[pp.second];
			auto start = SparseRREF::clocknow();
			if (thecol.size() > 1) {
				pool.detach_loop<index_t>(0, thecol.size(), [&](index_t j) {
					auto r = thecol[j];
					if (r == pp.first)
						return;
					auto entry = *sparse_mat_entry(mat, r, pp.second);
					sparse_vec_sub_mul(mat[r], mat[pp.first], entry, F);
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
		std::vector<std::vector<std::pair<index_t, index_t>>>& pivots,
		const field_t F, rref_option_t opt, int ordering) {
		std::vector<std::pair<index_t, index_t>> n_pivots;
		for (auto p : pivots)
			n_pivots.insert(n_pivots.end(), p.begin(), p.end());
		triangular_solver(mat, n_pivots, F, opt, ordering);
	}

	// dot product
	template <typename T, typename index_t>
	inline int sparse_mat_dot_sparse_vec(
		sparse_vec<index_t, T> result,
		const sparse_mat<T, index_t>& mat,
		const sparse_vec<index_t, T> vec, const field_t F) {
		result.zero();
		if (vec.nnz() == 0 || mat.nnz() == 0)
			return 0;

		for (size_t i = 0; i < mat.nrow; i++) {
			T tmp = sparse_vec_dot(mat[i], vec, F);
			if (tmp != 0)
				result.push_back(i, tmp);
		}
		return 1;
	}

	// A = B * C
	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_mul(
		const sparse_mat<T, index_t>& B, const sparse_mat<T, index_t>& C, 
		const field_t F, thread_pool* pool = nullptr) {

		sparse_mat<T, index_t> A(B.nrow, C.ncol);
		auto nthreads = 1;
		if (pool)
			nthreads = pool->get_thread_count();

		std::vector<T> cachedensedmat(A.ncol * nthreads);
		std::vector<SparseRREF::bit_array> nonzero_c(nthreads);
		for (size_t i = 0; i < nthreads; i++)
			nonzero_c[i].resize(A.ncol);

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
					if (!nonzero_c_vec.count(ind)) {
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

	// first write a stupid one
	template <typename T, typename index_t>
	void schur_complete(sparse_mat<T, index_t>& mat, index_t k, 
		const std::vector<std::pair<index_t, index_t>>& pivots,
		const field_t F, T* tmpvec, SparseRREF::bit_array& nonzero_c) {

		if (mat[k].nnz() == 0)
			return;

		// SparseRREF::bit_array nonzero_c(mat.ncol);
		nonzero_c.clear();

		for (auto [ind, val] : mat[k]) {
			nonzero_c.insert(ind);
			tmpvec[ind] = val;
		}

		ulong e_pr;
		for (auto [r, c] : pivots) {
			if (!nonzero_c.count(c))
				continue;
			T entry = tmpvec[c];
			if constexpr (std::is_same_v<T, ulong>) {
				e_pr = n_mulmod_precomp_shoup(tmpvec[c], F.mod.n);
			}
			for (auto [ind, val] : mat[r]) {
				if (!nonzero_c.count(ind)) {
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
				if (tmpvec[ind] == 0)
					nonzero_c.erase(ind);
			}
		}

		auto pos = nonzero_c.nonzero();
		mat[k].zero();
		for (auto p : pos) {
			mat[k].push_back(p, tmpvec[p]);
		}
	}

	// TODO: CHECK!!!
	// SLOW!!!
	template <typename T, typename index_t>
	void triangular_solver_2_rec(sparse_mat<T, index_t>& mat, 
		std::vector<std::vector<index_t>>& tranmat, 
		std::vector<std::pair<index_t, index_t>>& pivots,
		const field_t F, rref_option_t opt, T* cachedensedmat,
		std::vector<SparseRREF::bit_array>& nonzero_c, size_t n_split, size_t rank, size_t& process) {

		bool verbose = opt->verbose;
		auto& pool = opt->pool;
		opt->verbose = false;
		if (pivots.size() < n_split) {
			triangular_solver(mat, pivots, F, opt, -1);
			opt->verbose = verbose;
			process += pivots.size();
			return;
		}

		std::vector<std::pair<index_t, index_t>> sub_pivots(pivots.end() - n_split, pivots.end());
		std::vector<std::pair<index_t, index_t>> left_pivots(pivots.begin(), pivots.end() - n_split);

		std::unordered_set<index_t> pre_leftrows;
		for (auto [r, c] : sub_pivots)
			pre_leftrows.insert(tranmat[c].begin(), tranmat[c].end());
		for (auto [r, c] : sub_pivots)
			pre_leftrows.erase(r);
		std::vector<index_t> leftrows(pre_leftrows.begin(), pre_leftrows.end());

		// for printing
		size_t now_nnz = mat.nnz();
		int bitlen_nnz = (int)std::floor(std::log(now_nnz) / std::log(10)) + 3;
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

		if (verbose) {
			size_t old_cc = cc;
			while (cc < leftrows.size()) {
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
			}
		}

		pool.wait();

		triangular_solver(mat, sub_pivots, F, opt, -1);
		opt->verbose = verbose;
		process += sub_pivots.size();

		triangular_solver_2_rec(mat, tranmat, left_pivots, F, opt, cachedensedmat, nonzero_c, n_split, rank, process);
	}

	template <typename T, typename index_t>
	void triangular_solver_2(sparse_mat<T, index_t>& mat, std::vector<std::pair<index_t, index_t>>& pivots,
		const field_t F, rref_option_t opt) {

		auto& pool = opt->pool;
		// prepare the tmp array
		auto nthreads = pool.get_thread_count();
		std::vector<T> cachedensedmat(mat.ncol * nthreads);
		std::vector<SparseRREF::bit_array> nonzero_c(nthreads);
		for (size_t i = 0; i < nthreads; i++)
			nonzero_c[i].resize(mat.ncol);

		// we only need to compute the transpose of the submatrix involving pivots
		std::vector<std::vector<index_t>> tranmat(mat.ncol);
		for (size_t i = 0; i < pivots.size(); i++) {
			for (auto [col, val] : mat[pivots[i].first]) {
				if (val == 0)
					continue;
				tranmat[col].push_back(pivots[i].first);
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
		std::vector<std::vector<std::pair<index_t, index_t>>>& pivots,
		const field_t F, rref_option_t opt) {

		std::vector<std::pair<index_t, index_t>> n_pivots;
		// the first pivot is the row with only one nonzero value, so there is no need to do the elimination
		for (size_t i = 1; i < pivots.size(); i++)
			n_pivots.insert(n_pivots.end(), pivots[i].begin(), pivots[i].end());

		triangular_solver_2(mat, n_pivots, F, opt);
	}

	// TODO: TEST!!! 
	// TODO: add ordering
	// if already know the pivots, we can directly do the rref
	template <typename T, typename index_t>
	void sparse_mat_direct_rref(sparse_mat<T, index_t>& mat,
		const std::vector<std::vector<std::pair<index_t, index_t>>>& pivots, 
		const field_t F, rref_option_t opt) {
		auto& pool = opt->pool;

		// first set rows not in pivots to zero
		std::vector<index_t> rowset(mat.nrow, -1);
		size_t total_rank = 0;
		for (auto p : pivots) {
			total_rank += p.size();
			for (auto [r, c] : p)
				rowset[r] = c;
		}
		for (size_t i = 0; i < mat.nrow; i++)
			if (rowset[i] == -1)
				mat[i].zero();

		for (auto [r, c] : pivots[0]) {
			mat[r].zero();
			mat[r].push_back(c, 1);
			rowset[r] = -1;
		}

		std::vector<index_t> leftrows(mat.nrow, -1);
		eliminate_row_with_one_nnz(mat, leftrows, opt);

		leftrows.clear();

		// then do the elimination parallelly
		auto nthreads = pool.get_thread_count();
		std::vector<T> cachedensedmat(mat.ncol * nthreads);
		std::vector<SparseRREF::bit_array> nonzero_c(nthreads);
		for (size_t i = 0; i < nthreads; i++)
			nonzero_c[i].resize(mat.ncol);

		size_t rank = pivots[0].size();

		for (auto i = 1; i < pivots.size(); i++) {
			if (pivots[i].size() == 0)
				continue;

			// rescale the pivots
			for (auto [r, c] : pivots[i]) {
				T scalar = scalar_inv(*sparse_mat_entry(mat, r, c), F);
				sparse_vec_rescale(mat[r], scalar, F);
				rowset[r] = -1;
			}

			leftrows.clear();
			for (size_t j = 0; j < mat.nrow; j++) {
				if (rowset[j] != -1)
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
				}
				}, ((leftrows.size() < 20 * nthreads) ? 0 : leftrows.size() / 10));

			if (opt->verbose) {
				auto cn = clocknow();
				while (cc < leftrows.size()) {
					if (cc - old_cc > opt->print_step) {
						std::cout << "\r-- Row: "
							<< (int)std::floor(rank + (cc * 1.0 / leftrows.size()) * pivots[i].size())
							<< "/" << total_rank << "  nnz: " << mat.nnz()
							<< "  speed: " << ((1.0 * (cc - old_cc)) / usedtime(cn,clocknow()))
							<< " row/s          " << std::flush;
						old_cc = cc;
					}
				}
			}
			pool.wait();
			rank += pivots[i].size();
		}
		if (opt->verbose) {
			std::cout << std::endl;
		}
	}

	template <typename T, typename index_t>
	std::vector<std::vector<std::pair<index_t, index_t>>>
		sparse_mat_rref_c(sparse_mat<T, index_t>& mat, const field_t F, rref_option_t opt) {
		// first canonicalize, sort and compress the matrix

		auto& pool = opt->pool;
		auto nthreads = pool.get_thread_count();

		pool.detach_loop(0, mat.nrow, [&](auto i) { mat[i].compress(); });

		// perm the col
		std::vector<index_t> leftcols = perm_init((index_t)(mat.ncol));

		auto printstep = opt->print_step;
		bool verbose = opt->verbose;

		size_t now_nnz = mat.nnz();

		// store the pivots that have been used
		// -1 is not used
		std::vector<index_t> rowpivs(mat.nrow, -1);
		std::vector<std::vector<std::pair<index_t, index_t>>> pivots;
		std::vector<std::pair<index_t, index_t>> n_pivots;

		pool.wait();

		// look for row with only one non-zero entry

		// compute the transpose of pointers of the matrix
		size_t count = eliminate_row_with_one_nnz_rec(mat, rowpivs, opt);
		now_nnz = mat.nnz();

		for (size_t i = 0; i < mat.nrow; i++) {
			if (rowpivs[i] != -1)
				n_pivots.push_back(std::make_pair(i, rowpivs[i]));
		}
		pivots.push_back(n_pivots);

		sparse_mat<bool, index_t> tranmat(mat.ncol, mat.nrow);
		sparse_mat_transpose_replace(tranmat, mat, &pool);

		// sort pivots by nnz, it will be faster
		std::stable_sort(leftcols.begin(), leftcols.end(),
			[&tranmat](index_t a, index_t b) {
				return tranmat[a].nnz() < tranmat[b].nnz();
			});

		// look for pivot cols with only one nonzero element
		size_t kk = 0;
		n_pivots.clear();
		for (; kk < mat.ncol; kk++) {
			auto nnz = tranmat[leftcols[kk]].nnz();
			if (nnz == 0)
				continue;
			if (nnz == 1) {
				auto row = tranmat[leftcols[kk]](0);
				if (rowpivs[row] != -1)
					continue;
				if (opt->col_weight(leftcols[kk]) < 0)
					continue;
				rowpivs[row] = leftcols[kk];
				T scalar = scalar_inv(*sparse_mat_entry(mat, row, rowpivs[row]), F);
				sparse_vec_rescale(mat[row], scalar, F);
				n_pivots.push_back(std::make_pair(row, leftcols[kk]));
			}
			else if (nnz > 1)
				break; // since it's sorted
		}
		leftcols.erase(leftcols.begin(), leftcols.begin() + kk);
		pivots.push_back(n_pivots);
		auto rank = pivots[0].size() + pivots[1].size();

		std::vector<T> cachedensedmat(mat.ncol * nthreads);
		std::vector<SparseRREF::bit_array> nonzero_c(nthreads);
		for (size_t i = 0; i < nthreads; i++)
			nonzero_c[i].resize(mat.ncol);

		std::vector<index_t> leftrows;
		leftrows.reserve(mat.nrow);
		for (size_t i = 0; i < mat.nrow; i++) {
			if (rowpivs[i] != -1 || mat[i].nnz() == 0)
				continue;
			leftrows.push_back(i);
		}

		// for printing
		double oldpr = 0;
		int bitlen_nnz = (int)std::floor(std::log(now_nnz) / std::log(10)) + 3;
		int bitlen_ncol = (int)std::floor(std::log(mat.ncol) / std::log(10)) + 1;

		bit_array tmp_set(mat.ncol);

		while (kk < mat.ncol) {
			auto start = SparseRREF::clocknow();

			auto ps = findmanypivots(mat, tranmat, rowpivs, leftcols, opt->col_weight);
			if (ps.size() == 0)
				break;

			n_pivots.clear();
			for (auto i = ps.rbegin(); i != ps.rend(); i++) {
				rowpivs[(*i).first] = (*i).second;
				n_pivots.push_back(*i);
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

			std::vector<int> flags(leftrows.size(), 0);
			pool.detach_blocks<size_t>(0, leftrows.size(), [&](const size_t s, const size_t e) {
				auto id = SparseRREF::thread_id();
				for (size_t i = s; i < e; i++) {
					schur_complete(mat, leftrows[i], n_pivots, F, cachedensedmat.data() + id * mat.ncol, nonzero_c[id]);
					flags[i] = 1;
				}
				}, (leftrows.size() < 20 * nthreads ? 0 : leftrows.size() / 10));

			// remove used cols
			size_t localcount = 0;
			tmp_set.clear();
			for (auto [r, c] : ps)
				tmp_set.insert(c);
			for (auto it : leftcols) {
				if (!tmp_set.count(it)) {
					leftcols[localcount] = it;
					localcount++;
					tranmat[it].zero();
				}
				tranmat[it].clear();
			}
			leftcols.resize(localcount);

			bool print_once = true; // print at least once

			localcount = 0;
			while (localcount < leftrows.size()) {
				for (size_t i = 0; i < leftrows.size() && localcount < leftrows.size(); i++) {
					if (flags[i]) {
						auto row = leftrows[i];
						for (size_t j = 0; j < mat[row].nnz(); j++) {
							tranmat[mat[row](j)].push_back(row, true);
						}
						flags[i] = 0;
						localcount++;
					}
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

			kk += ps.size();
		}

		if (verbose)
			std::cout << "\n** Rank: " << rank << " nnz: " << mat.nnz() << std::endl;

		return pivots;
	}

	template <typename T, typename index_t>
	std::vector<std::vector<std::pair<index_t, index_t>>> 
		sparse_mat_rref(sparse_mat<T, index_t>& mat, const field_t F, rref_option_t opt) {

		auto pivots = sparse_mat_rref_c(mat, F, opt);

		if (opt->shrink_memory) {
			opt->pool.detach_loop(0, mat.nrow, [&](auto i) {
					mat[i].reserve(mat[i].nnz());
				});
			opt->pool.wait();
		}

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
	std::vector<std::vector<std::pair<index_t, index_t>>> sparse_mat_rref_reconstruct(
		sparse_mat<rat_t, index_t>& mat, rref_option_t opt, const bool checkrank = false) {
		std::vector<std::vector<std::pair<index_t, index_t>>> pivots;

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

		pivots = sparse_mat_rref_c(matul, F, opt);

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

		ulong old_height = matq.height_bits();

		// set rows not in pivots to zero
		if (!isok || mod.bits() < mat_height_bits) {
			std::vector<index_t> rowset(mat.nrow, -1);
			for (auto p : pivots)
				for (auto [r, c] : p)
					rowset[r] = c;
			for (size_t i = 0; i < mat.nrow; i++)
				if (rowset[i] == -1)
					mat[i].zero();
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
		const std::vector<std::pair<index_t, index_t>>& pivots, field_t F, rref_option_t opt) {

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
			rows[i] = M[pivots[i].first];
		}
		sparse_mat_transpose_replace(trows, rows);

		std::vector<index_t> colpivs(M.ncol, -1);
		std::vector<index_t> nonpivs;
		for (size_t i = 0; i < rank; i++)
			colpivs[pivots[i].second] = pivots[i].first;

		for (auto i = 0; i < M.ncol; i++)
			if (colpivs[i] == -1)
				nonpivs.push_back(i);

		K.init(M.ncol - rank, M.ncol);
		pool.detach_loop<size_t>(0, nonpivs.size(), [&](size_t i) {
			auto& thecol = trows[nonpivs[i]];
			K[i].reserve(thecol.nnz() + 1);
			for (size_t j = 0; j < thecol.nnz(); j++) {
				K[i].push_back(pivots[thecol(j)].second, thecol[j]);
			}
			K[i].push_back(nonpivs[i], m1);
			K[i].sort_indices();
			});
		pool.wait();

		return K;
	}

	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_rref_kernel(const sparse_mat<T, index_t>& M,
		const std::vector<std::vector<std::pair<index_t, index_t>>>& pivots, 
		const field_t F, rref_option_t opt) {
		std::vector<std::pair<index_t, index_t>> n_pivots;
		for (auto& p : pivots)
			n_pivots.insert(n_pivots.end(), p.begin(), p.end());
		return sparse_mat_rref_kernel(M, n_pivots, F, opt);
	}

	template <typename T, typename index_t>
	sparse_mat<T, index_t> sparse_mat_inverse(const sparse_mat<T, index_t>& M, 
		const field_t F, rref_option_t opt) {
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
			if (i < M.nrow)
				return old_col_weight(i);
			else
				return -1LL;
			};
		opt->col_weight = new_col_weight;
		bool is_back_sub = opt->is_back_sub;
		opt->is_back_sub = true;

		std::vector<std::vector<std::pair<index_t, index_t>>> pivots;
		if (F.ring == RING::FIELD_Fp)
			pivots = sparse_mat_rref(M1, F, opt);
		else if (F.ring == RING::FIELD_QQ)
			pivots = sparse_mat_rref_reconstruct(M1, opt, true);
		else {
			std::cerr << "Error: sparse_mat_inverse: field not supported" << std::endl;
			return sparse_mat<T, index_t>();
		}

		std::vector<std::pair<index_t, index_t>> flatten_pivots;
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
				return flatten_pivots[a].second < flatten_pivots[b].second;
			});
		for (size_t i = 0; i < M.nrow; i++) {
			perm[i] = flatten_pivots[perm[i]].first;
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
	sparse_mat<ScalarType, index_t> sparse_mat_read(T& st, const field_t F) {
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
	sparse_mat<T, index_t> sparse_mat_read_wxf(const WXF_PARSER::ExprTree& tree, const field_t F) {
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
			for (size_t i = 0; i < nnz; i++) {
				vals[i] = ptr[i];
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
		case SPARSE_FILE_TYPE_PLAIN: {
			st << mat.nrow << ' ' << mat.ncol << ' ' << mat.nnz() << '\n';
			break;
		}
		case SPARSE_FILE_TYPE_MTX: {
			if constexpr (std::is_same_v<T, ulong>) {
				st << "%%MatrixMarket matrix coordinate integer general\n";
			}
			st << mat.nrow << ' ' << mat.ncol << ' ' << mat.nnz() << '\n';
			break;
		}
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

		for (size_t i = 0; i < mat.nrow; ++i) {
			for (auto [ind, val] : mat[i]) {
				if (val == 0) {
					continue;
				}
				auto [ptr1, ec1] = std::to_chars(num_buf, num_buf + sizeof(num_buf), i + 1);
				st.write(num_buf, ptr1 - num_buf);
				st.put(' ');

				auto [ptr2, ec2] = std::to_chars(num_buf, num_buf + sizeof(num_buf), ind + 1);
				st.write(num_buf, ptr2 - num_buf);
				st.put(' ');

				st << val;
				st.put('\n');
			}
		}

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