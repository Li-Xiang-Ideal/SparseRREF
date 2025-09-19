/*
	Copyright (C) 2024 Zhenjie Li (Li, Zhenjie)

	This file is part of Sparse_rref. The Sparse_rref is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

/*
	To compile the library, WolframLibrary.h and WolframSparseLibrary.h are required,
	which are included in the Mathematica installation directory.

	The output of modrref/ratrref is the join of the rref of A and its kernel.

	To load it in Mathematica, use the following code (as an example):

	```mathematica

	modrreflib = 
	LibraryFunctionLoad[
		"rreflib.dll",
		"modrref", {{LibraryDataType[SparseArray], 
		  "Constant"}, {Integer}, {Integer}, {Integer}}, {LibraryDataType[
		  SparseArray], Automatic}];

	ratrreflib = LibraryFunctionLoad[
		"rreflib.dll",
		"rational_rref", {{LibraryDataType[ByteArray],
		  "Constant"}, {Integer}, {Integer}}, {LibraryDataType[ByteArray],
		 Automatic}];
	
	(* the first matrix is the result of rref, and the second is its kernel *)
	modprref[mat_SparseArray, p_Integer, method_ : 1, nthread_ : 1] := 
		With[{joinedmat = modrreflib[mat, p, method, nthread]},
		 If[method =!= 0, {joinedmat[[;; Length@mat]], 
		   Transpose[joinedmat[[Length@mat + 1 ;;]]]}, joinedmat]];

	ratrref[mat_SparseArray, mode_ : 1, nthread_ : 1] := 
		BinaryDeserialize[ratrreflib[BinarySerialize[mat], mode, nthread]];

	```
*/

#include <string>
#include "sparse_mat.h"
#include "wxf_support.h"
#include "mma/WolframLibrary.h"
#include "mma/WolframSparseLibrary.h"
#include "mma/WolframNumericArrayLibrary.h"

using namespace SparseRREF;

sparse_mat<ulong> MSparseArray_to_sparse_mat_ulong(WolframLibraryData ld, MArgument* arg, ulong p) {
	auto mat = MArgument_getMSparseArray(*arg);
	auto sf = ld->sparseLibraryFunctions;

	auto dims = sf->MSparseArray_getDimensions(mat);
	auto nrows = dims[0];
	auto ncols = dims[1];

	auto m_rowptr = sf->MSparseArray_getRowPointers(mat);
	auto m_colptr = sf->MSparseArray_getColumnIndices(mat);
	auto m_valptr = sf->MSparseArray_getExplicitValues(mat);

	// rowptr, valptr, colptr are managed by mathematica
	// do not free them
	mint* rowptr = ld->MTensor_getIntegerData(*m_rowptr);
	mint* valptr = ld->MTensor_getIntegerData(*m_valptr);
	mint* colptr = ld->MTensor_getIntegerData(*m_colptr);

	auto nnz = rowptr[nrows];

	// init a sparse matrix
	nmod_t pp;
	int_t tmp;
	nmod_init(&pp, (ulong)p);
	sparse_mat<ulong> A(nrows, ncols);

	for (auto i = 0; i < nrows; i++) {
		A[i].reserve(rowptr[i + 1] - rowptr[i]);
		for (auto k = rowptr[i]; k < rowptr[i + 1]; k++) {
			tmp = valptr[k];
			A[i].push_back(colptr[k] - 1, tmp % pp);
		}
	}

	return A;
}

int sparse_mat_ulong_to_MSparseArray(WolframLibraryData ld, MSparseArray& res, const sparse_mat<ulong>& A) {
	auto sf = ld->sparseLibraryFunctions;
	size_t nnz = A.nnz();
	MTensor pos, val, dim;
	mint dims_r2[] = { nnz, 2 };
	ld->MTensor_new(MType_Integer, 2, dims_r2, &pos);
	mint dims_r1[] = { nnz };
	ld->MTensor_new(MType_Integer, 1, dims_r1, &val);
	dims_r1[0] = 2;
	ld->MTensor_new(MType_Integer, 1, dims_r1, &dim);
	mint* dimdata = ld->MTensor_getIntegerData(dim);
	dimdata[0] = A.nrow;
	dimdata[1] = A.ncol;
	mint* valdata = ld->MTensor_getIntegerData(val);
	mint* posdata = ld->MTensor_getIntegerData(pos);
	auto nownnz = 0;
	for (size_t i = 0; i < A.nrow; i++) {
		for (size_t j = 0; j < A[i].nnz(); j++) {
			posdata[2 * nownnz] = i + 1;
			posdata[2 * nownnz + 1] = A[i](j) + 1;
			valdata[nownnz] = A[i][j];
			nownnz++;
		}
	}
	auto err = sf->MSparseArray_fromExplicitPositions(pos, val, dim, 0, &res);
	ld->MTensor_free(pos);
	ld->MTensor_free(val);
	ld->MTensor_free(dim);
	return err;
}

EXTERN_C DLLEXPORT int modpmatmul(WolframLibraryData ld, mint Argc, MArgument* Args, MArgument Res) {
	if (Argc != 4)
		return LIBRARY_FUNCTION_ERROR;
	auto matA = MArgument_getMSparseArray(Args[0]);
	auto matB = MArgument_getMSparseArray(Args[1]);
	auto p = MArgument_getInteger(Args[2]);
	auto nthreads = MArgument_getInteger(Args[3]);
	auto sf = ld->sparseLibraryFunctions;
	auto ranksA = sf->MSparseArray_getRank(matA);
	if (ranksA != 2 && sf->MSparseArray_getImplicitValue(matA) != 0)
		return LIBRARY_FUNCTION_ERROR;
	auto ranksB = sf->MSparseArray_getRank(matB);
	if (ranksB != 2 && sf->MSparseArray_getImplicitValue(matB) != 0)
		return LIBRARY_FUNCTION_ERROR;
	auto A = MSparseArray_to_sparse_mat_ulong(ld, Args, (ulong)p);
	auto B = MSparseArray_to_sparse_mat_ulong(ld, Args + 1, (ulong)p);
	if (A.ncol != B.nrow)
		return LIBRARY_FUNCTION_ERROR;
	field_t F(FIELD_Fp, p);
	int err = 0;
	MSparseArray result = 0;
	if (nthreads == 1) {
		auto C = sparse_mat_mul(A, B, F, nullptr);
		err = sparse_mat_ulong_to_MSparseArray(ld, result, C);
	}
	else {
		thread_pool pool(nthreads);
		auto C = sparse_mat_mul(A, B, F, &pool);
		err = sparse_mat_ulong_to_MSparseArray(ld, result, C);
	}
	if (err)
		return LIBRARY_FUNCTION_ERROR;
	MArgument_setMSparseArray(Res, result);
	return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int modrref(WolframLibraryData ld, mint Argc, MArgument *Args, MArgument Res) {
	if (Argc != 4)
		return LIBRARY_FUNCTION_ERROR;
	auto mat = MArgument_getMSparseArray(Args[0]);
	auto p = MArgument_getInteger(Args[1]);
	auto output_kernel = MArgument_getInteger(Args[2]);
	auto nthreads = MArgument_getInteger(Args[3]);

	auto sf = ld->sparseLibraryFunctions;

	auto ranks = sf->MSparseArray_getRank(mat);
	if (ranks != 2 && sf->MSparseArray_getImplicitValue(mat) != 0)
		return LIBRARY_FUNCTION_ERROR;
	

	auto A = MSparseArray_to_sparse_mat_ulong(ld, Args, (ulong)p);

	field_t F(FIELD_Fp, p);

	rref_option_t opt;
	opt->pool.reset(nthreads);

	auto pivots = sparse_mat_rref(A, F, opt);
	sparse_mat<ulong> K;
	size_t len = 0;
	if (output_kernel) {
		K = sparse_mat_rref_kernel(A, pivots, F, opt);
		K = K.transpose();
		len = K.nrow;
	} 

	A.append(std::move(K));

	MSparseArray result = 0;
	int err = sparse_mat_ulong_to_MSparseArray(ld, result, A);
	
	if (err)
		return LIBRARY_FUNCTION_ERROR;

	MArgument_setMSparseArray(Res, result);
	// MArgument_setMTensor(Res, pos);

	return LIBRARY_NO_ERROR;
}

// output_mode:
// 0: output the rref
// 1: output the rref and its kernel
// 2: output the rref and its pivots
// 3: output the rref, kernel and pivots
EXTERN_C DLLEXPORT int rational_rref(WolframLibraryData ld, mint Argc, MArgument* Args, MArgument Res) {
	if (Argc != 3)
		return LIBRARY_FUNCTION_ERROR;
	auto na_in = MArgument_getMNumericArray(Args[0]);
	auto output_mode = MArgument_getInteger(Args[1]);
	auto nthreads = MArgument_getInteger(Args[2]);

	numericarray_data_t type = MNumericArray_Type_Undef;
	auto naFuns = ld->numericarrayLibraryFunctions;

	type = naFuns->MNumericArray_getType(na_in);
	if (type != MNumericArray_Type_UBit8)
		return LIBRARY_FUNCTION_ERROR;

	auto in_str = (uint8_t*)(naFuns->MNumericArray_getData(na_in));
	auto length = naFuns->MNumericArray_getFlattenedLength(na_in);

	std::vector<uint8_t> res_str;
	uint8_t* out_str = nullptr;
	mint out_len = 0;
	auto err = LIBRARY_NO_ERROR;
	MNumericArray na_out = NULL;
	{
		field_t F(FIELD_QQ);

		WXF_PARSER::ExprTree expr_tree;
		expr_tree = WXF_PARSER::MakeExprTree(in_str, (size_t)length);
		auto mat = sparse_mat_read_wxf<rat_t, int>(expr_tree, F);

		rref_option_t opt;
		opt->pool.reset(nthreads);

		std::atomic<bool> cancel(false);
		std::thread check_cancel([&]() {
			while (!cancel) {
				cancel = ld->AbortQ();
				opt->abort = cancel.load();
				// wait for 100 ms before checking again
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}
			});

		auto pivots = sparse_mat_rref_reconstruct(mat, opt);
		std::vector<pivot_t<int>> pivots_vec;
		for (auto& p : pivots) {
			pivots_vec.insert(pivots_vec.end(), p.begin(), p.end());
		}

		sparse_mat<rat_t> K;
		size_t len = 0;
		if (output_mode == 1 || output_mode == 3) {
			K = sparse_mat_rref_kernel(mat, pivots, F, opt);
			len = K.nrow;
		}

		{
			using namespace WXF_PARSER;

			std::vector<uint8_t> m_str;
			auto push_mat = [&](const auto& M) {
				m_str = sparse_mat_write_wxf(M, false);
				res_str.insert(res_str.end(), m_str.begin(), m_str.end());
				};
			auto push_null = [&]() {
				res_str.push_back((uint8_t)WXF_PARSER::WXF_HEAD::symbol);
				res_str.push_back(4); // length of Null
				std::string null_str = "Null";
				res_str.insert(res_str.end(), null_str.begin(), null_str.end());
				};
			auto push_pivots = [&]() {
				// rank 2, dimensions {pivots_vec.size(), 2}
				TOKEN token(WXF_HEAD::array, { pivots_vec.size(), 2 }, 3, 2 * pivots_vec.size(), false);
				token.to_ustr(res_str);
				uint8_t int64_buf[16];
				for (auto& p : pivots_vec) {
					// output the pivot position
					int64_t row = p.r + 1; // 1-based index
					int64_t col = p.c + 1; // 1-based index
					memcpy(int64_buf, &row, sizeof(row));
					memcpy(int64_buf + sizeof(row), &col, sizeof(col));
					res_str.insert(res_str.end(), int64_buf, int64_buf + sizeof(row) + sizeof(col));
				}
				};
			auto push_list = [&](size_t n) {
				res_str.push_back(56); res_str.push_back(58);
				// function, List, n
				TOKEN(WXF_HEAD::func, n).to_ustr(res_str);
				TOKEN(WXF_HEAD::symbol, "List").to_ustr(res_str);
				};
			
			switch (output_mode) {
			case 0: // output the rref
				res_str = sparse_mat_write_wxf(mat, true);
				break;
			case 1: {// output the rref and its kernel
				push_list(2);
				push_mat(mat);
				if (len > 0) 
					push_mat(K);
				else 
					push_null();
				break;
			}
			case 2: { // output the rref and its pivots
				push_list(2);
				push_mat(mat);
				push_pivots();
				break;
			}
			case 3: { // output the rref, kernel and pivots
				push_list(3);
				push_mat(mat);
				if (len > 0)
					push_mat(K);
				else
					push_null();
				push_pivots();
				break;
			}
			default:
				return LIBRARY_FUNCTION_ERROR;
			}
		}

		cancel = true;
		check_cancel.join();

		if (opt->abort) {
			return LIBRARY_FUNCTION_ERROR;
		}

		// output the result(bit_array)
		out_len = res_str.size();
		auto err = naFuns->MNumericArray_new(type, 1, &out_len, &na_out);

		if (err)
			return LIBRARY_FUNCTION_ERROR;

		out_str = (uint8_t*)(naFuns->MNumericArray_getData(na_out));
	}

	std::memcpy(out_str, res_str.data(), res_str.size() * sizeof(uint8_t));

	MArgument_setMNumericArray(Res, na_out);

	return err;
}

EXTERN_C DLLEXPORT int ratmat_inv(WolframLibraryData ld, mint Argc, MArgument* Args, MArgument Res) {
	if (Argc != 2)
		return LIBRARY_FUNCTION_ERROR;
	auto na_in = MArgument_getMNumericArray(Args[0]);
	auto nthreads = MArgument_getInteger(Args[1]);

	numericarray_data_t type = MNumericArray_Type_Undef;
	auto naFuns = ld->numericarrayLibraryFunctions;

	type = naFuns->MNumericArray_getType(na_in);
	if (type != MNumericArray_Type_UBit8)
		return LIBRARY_FUNCTION_ERROR;

	auto in_str = (uint8_t*)(naFuns->MNumericArray_getData(na_in));
	auto length = naFuns->MNumericArray_getFlattenedLength(na_in);

	std::vector<uint8_t> res_str;
	uint8_t* out_str = nullptr;
	mint out_len = 0;
	auto err = LIBRARY_NO_ERROR;
	MNumericArray na_out = NULL;
	{
		field_t F(FIELD_QQ);

		WXF_PARSER::ExprTree expr_tree;
		expr_tree = WXF_PARSER::MakeExprTree(in_str, (size_t)length);
		auto mat = sparse_mat_read_wxf<rat_t, int>(expr_tree, F);

		rref_option_t opt;
		opt->pool.reset(nthreads);

		auto inv_mat = sparse_mat_inverse(mat, F, opt);
		res_str = sparse_mat_write_wxf(inv_mat, true);

		// output the result(bit_array)
		out_len = res_str.size();
		auto err = naFuns->MNumericArray_new(type, 1, &out_len, &na_out);

		if (err)
			return LIBRARY_FUNCTION_ERROR;

		out_str = (uint8_t*)(naFuns->MNumericArray_getData(na_out));
	}

	std::memcpy(out_str, res_str.data(), res_str.size() * sizeof(uint8_t));

	MArgument_setMNumericArray(Res, na_out);

	return err;
}