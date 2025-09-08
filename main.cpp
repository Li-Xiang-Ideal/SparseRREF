/*
	Copyright (C) 2024 Zhenjie Li (Li, Zhenjie)

	This file is part of SparseRREF. The SparseRREF is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

#include <filesystem>
#include <fstream>
#include <iostream>

// Use mimalloc for memory management or not
// #define USE_MIMALLOC 1

#include "argparse.hpp"
#include "sparse_mat.h"

#ifdef _WIN32
#include <conio.h>
int getch_key() {
	if (_kbhit())
		return _getch();
	else
		return -1;
}
#else
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

int getch_key() {
	char buf = 0;
	struct termios old = {};
	if (tcgetattr(STDIN_FILENO, &old) < 0) return -1;

	struct termios new_t = old;
	new_t.c_lflag &= ~(ICANON | ECHO);  
	new_t.c_iflag &= ~IXON;             
	if (tcsetattr(STDIN_FILENO, TCSANOW, &new_t) < 0) return -1;

	int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
	if (flags == -1) return -1;
	fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);

	int nread = read(STDIN_FILENO, &buf, 1);

	fcntl(STDIN_FILENO, F_SETFL, flags);
	tcsetattr(STDIN_FILENO, TCSADRAIN, &old);

	if (nread > 0)
		return static_cast<int>(buf);
	else
		return -1; 
}
#endif

void key_listener(std::atomic<bool>& stop_flag) {
	while (!stop_flag) {
		int c = getch_key();
		if (c == 17) { // Ctrl+Q
			std::cout << "\n[Ctrl+Q] pressed. Stopping...\n";
			stop_flag = true;
			break;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	}
}

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
	program.add_argument("-m", "--method")
		.help("method of RREF ")
		.default_value(0)
		.nargs(1)
		.scan<'i', int>();
	program.add_argument("-op", "--output-pivots")
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
	program.add_argument("-nb", "--no-backward-substitution")
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
	opt->method = program.get<int>("--method");
	if (nthread == 0)
		opt->pool.reset(); // automatic mode, use all possible threads
	else
		opt->pool.reset(nthread);

	std::thread thread_listener(key_listener, std::ref(opt->abort));

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
	
	using index_t = int;
	std::variant<sparse_mat<rat_t, index_t>, sparse_mat<ulong, index_t>> mat;

	auto file_ext = filePath.extension().string();
	if (file_ext == ".wxf" || file_ext == ".WXF") {
		if (prime == 0)
			mat = sparse_mat_read_wxf<rat_t, index_t>(filePath, F);
		else
			mat = sparse_mat_read_wxf<ulong, index_t>(filePath, F);
	}
	else {
		std::ifstream file(filePath);

		if (prime == 0)
			mat = sparse_mat_read<rat_t, index_t>(file, F);
		else
			mat = sparse_mat_read<ulong, index_t>(file, F);
	}

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
	std::vector<std::vector<pivot_t<index_t>>> pivots;
	if (prime == 0) {
		pivots = sparse_mat_rref_reconstruct(std::get<0>(mat), opt);
	}
	else {
		pivots = sparse_mat_rref(std::get<1>(mat), F, opt);
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
			for (auto [r, c] : p)
				file2 << r + 1 << ", " << c + 1 << '\n';
		}
		file2.close();
	}

	if (outname == input_file)
		outname_add = ".rref";
	else
		outname_add = "";

	if (file_ext == ".wxf" || file_ext == ".WXF") {
		std::vector<uint8_t> wxfdata;
		if (prime == 0)
			wxfdata = sparse_mat_write_wxf(std::get<0>(mat));
		else
			wxfdata = sparse_mat_write_wxf(std::get<1>(mat));

		auto parent_path = filePath.parent_path();
		auto outname_path = parent_path / (outname + outname_add);
		ustr_write(outname_path, wxfdata);
	}
	else {
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
	}

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

	opt->abort = true; // stop the key listener

	thread_listener.join();

	return 0;
}