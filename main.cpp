/*
	Copyright (C) 2024 Zhenjie Li (Li, Zhenjie)

	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

#include <filesystem>
#include <fstream>
#include <iostream>

#include "argparse.hpp"
#include "sparse_mat.h"

using namespace SparseRREF;

#define printtime(str)                                                         \
    std::cout << (str) << " spent " << std::fixed << std::setprecision(6)      \
              << SparseRREF::usedtime(start, end) << " seconds." << std::endl

#define printmatinfo(mat)                                                      \
    std::cout << "nnz: " <<  (mat).nnz() << " ";                               \
    std::cout << "nrow: " << (mat).nrow << " ";                                \
    std::cout << "ncol: " << (mat).ncol << std::endl

int main(int argc, char** argv) {
	argparse::ArgumentParser program("SparseRREF", SparseRREF::version);
	program.set_usage_max_line_width(80);
	program.add_description("(exact) Sparse Reduced Row Echelon Form " + std::string(SparseRREF::version));
	program.add_argument("input_file")
		.help("input file in the Matrix Market exchange formats (MTX) or\nSparse/Symbolic Matrix Storage (SMS)");
	program.add_argument("-o", "--output")
		.help("output file in MTX format")
		.default_value("<input_file>.rref")
		.nargs(1);
	program.add_usage_newline();
	program.add_argument("-k", "--kernel")
		.default_value(false)
		.help("output the kernel (null vectors)")
		.implicit_value(true)
		.nargs(0);
	program.add_argument("--output-pivots")
		.help("output pivots")
		.default_value(false)
		.implicit_value(true)
		.nargs(0);
	program.add_usage_newline();
	program.add_argument("-F", "--field")
		.default_value("QQ")
		.help("QQ: rational field\nZp or Fp: Z/p for a prime p")
		.nargs(1);
	program.add_argument("-p", "--prime")
		.default_value("34534567")
		.help("a prime number, only vaild when field is Zp ")
		.nargs(1);
	program.add_argument("-t", "--threads")
		.help("the number of threads ")
		.default_value(1)
		.nargs(1)
		.scan<'i', int>();
	program.add_usage_newline();
	program.add_argument("-V", "--verbose")
		.default_value(false)
		.help("prints information of calculation")
		.implicit_value(true)
		.nargs(0);
	program.add_argument("-ps", "--print_step")
		.default_value(100)
		.help("print step when --verbose is enabled")
		.nargs(1)
		.scan<'i', int>();
	program.add_usage_newline();
	program.add_argument("--no-backward-substitution")
		.help("no backward substitution")
		.default_value(false)
		.implicit_value(true)
		.nargs(0);

	try {
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err) {
		std::cerr << err.what() << std::endl;
		std::cerr << program;
		std::exit(1);
	}

	ulong prime;
	if (program.get<std::string>("--field") == "QQ") {
		prime = 0;
		std::cout << "RREF on the rational field. Using the reconstruction method." << std::endl;
	}
	else if (program.get<std::string>("--field") == "Zp" || program.get<std::string>("--field") == "Fp") {
		if (program.get<std::string>("--prime") == "34534567") {
			prime = 34534567;
		}
		else {
			auto str = program.get<std::string>("--prime");
			int_t prep(str);
			if (prep > (1ULL << ((FLINT64) ? 63 : 31))) {
				std::cerr << "The prime number is too large: " << str
					<< std::endl;
				std::cerr << "It should be less than " << 2 << "^"
					<< ((FLINT64) ? 63 : 31) << std::endl;
				std::exit(1);
			}
			prime = prep.to_ui();
			if (!n_is_prime(prime)) {
				prime = n_nextprime(prime - 1, 0);
				std::cerr
					<< "The number is not a prime, use a near prime instead."
					<< std::endl;
			}
		}
		std::cout << "Using prime: " << prime << std::endl;
	}
	else {
		std::cerr << "The field is not valid: "
			<< program.get<std::string>("--field") << std::endl;
		return 1;
	}

	rref_option_t opt;
	int nthread = program.get<int>("--threads");
	if (nthread == 0)
		opt->pool.reset(); // automatic mode, use all possible threads
	else
		opt->pool.reset(nthread);

	std::cout << "using " << nthread << " threads" << std::endl;

	field_t F;
	if (prime == 0)
		F = field_t(SparseRREF::RING::FIELD_QQ);
	else
		F = field_t(SparseRREF::RING::FIELD_Fp, prime);

	auto start = SparseRREF::clocknow();
	auto input_file = program.get<std::string>("input_file");
	std::filesystem::path filePath = input_file;
	if (!std::filesystem::exists(filePath)) {
		std::cerr << "File does not exist: " << filePath << std::endl;
		return 1;
	}

	std::ifstream file(filePath);

	std::variant<sparse_mat<rat_t>, sparse_mat<ulong>> mat;

	if (prime == 0) 
		mat = sparse_mat_read<rat_t>(file, F);
	else 
		mat = sparse_mat_read<ulong>(file, F);

	file.close();

	auto end = SparseRREF::clocknow();
	std::cout << "-------------------" << std::endl;
	printtime("read");

	if (prime == 0) {
		printmatinfo(std::get<0>(mat));
	}
	else {
		printmatinfo(std::get<1>(mat));
	}

	opt->verbose = (program["--verbose"] == true);
	opt->is_back_sub = (program["--no-backward-substitution"] == false);
	opt->print_step = program.get<int>("--print_step");

	if (opt->verbose) {
		std::cout << "-------------------" << std::endl;
		std::cout << ">> RREFing: " << std::endl;
	}

	start = SparseRREF::clocknow();
	std::vector<std::vector<std::pair<slong, slong>>> pivots;
	if (prime == 0) {
		pivots = sparse_mat_rref_reconstruct(std::get<sparse_mat<rat_t>>(mat), opt);
	}
	else {
		pivots = sparse_mat_rref(std::get<sparse_mat<ulong>>(mat), F, opt);
	}

	end = SparseRREF::clocknow();
	std::cout << "-------------------" << std::endl;
	printtime("RREF");

	size_t rank = 0;
	for (auto p : pivots) {
		rank += p.size();
	}
	std::cout << "rank: " << rank << " ";
	if (prime == 0) {
		printmatinfo(std::get<0>(mat));
	}
	else {
		printmatinfo(std::get<1>(mat));
	}

	start = SparseRREF::clocknow();
	std::ofstream file2;
	std::string outname, outname_add("");
	if (program.get<std::string>("--output") == "<input_file>.rref")
		outname = input_file;
	else
		outname = program.get<std::string>("--output");

	if (program["--output-pivots"] == true) {
		outname_add = ".piv";
		file2.open(outname + outname_add);
		for (auto p : pivots) {
			for (auto ii : p)
				file2 << ii.first + 1 << ", " << ii.second + 1 << '\n';
		}
		file2.close();
	}

	if (outname == input_file)
		outname_add = ".rref";
	else
		outname_add = "";

	file2.open(outname + outname_add);
	if (prime == 0) {
		sparse_mat_write(std::get<0>(mat), file2, SparseRREF::SPARSE_FILE_TYPE_PLAIN);
	}
	else {
		if (prime > (1ULL << 32)) {
			std::cout << "Warning: the prime is too large, use plain format." << std::endl;
			sparse_mat_write(std::get<1>(mat), file2, SparseRREF::SPARSE_FILE_TYPE_PLAIN);
		}
		else
			sparse_mat_write(std::get<1>(mat), file2, SparseRREF::SPARSE_FILE_TYPE_MTX);
	}
	file2.close();

	if (program["--kernel"] == true) {
		outname_add = ".kernel";
		file2.open(outname + outname_add);
		if (prime == 0) {
			auto K = sparse_mat_rref_kernel(std::get<0>(mat), pivots, F, opt);
			if (K.nrow > 0)
				sparse_mat_write(K, file2, SparseRREF::SPARSE_FILE_TYPE_PLAIN);
			else
				std::cout << "Kernel is empty." << std::endl;
		}
		else {
			auto K = sparse_mat_rref_kernel(std::get<1>(mat), pivots, F, opt);
			if (K.nrow > 0) {
				if (prime > (1ULL << 32))
					sparse_mat_write(K, file2, SparseRREF::SPARSE_FILE_TYPE_PLAIN);
				else
					sparse_mat_write(K, file2, SparseRREF::SPARSE_FILE_TYPE_MTX);
			}
			else
				std::cout << "Kernel is empty." << std::endl;
		}
		file2.close();
	}

	end = SparseRREF::clocknow();
	printtime("write files");

	return 0;
}