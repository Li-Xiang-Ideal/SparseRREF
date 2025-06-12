#include <iostream>
#include <string>

#include "spasm.h"

int main(int argc, char **argv){
	if(argc != 3){
		std::cerr << "Usage: " << argv[0] << " prime <matrix.mtx>" << std::endl;
		return 1;
	}
	ulong p = atoll(argv[1]);
	FILE *f = fopen(argv[2], "r");
	if(f == NULL){
		std::cerr << "Error opening file" << std::endl;
		return 1;
	}
	auto m1 = spasm_triplet_load(f, p, NULL);
	auto m2 = spasm_compress(m1);
	spasm_triplet_free(m1);
	int n = m2->n;
	int m = m2->m;
	fclose(f);

	struct echelonize_opts dopts;
	auto opts = &dopts;
	// std::cout << "[echelonize] using default settings\n";
	spasm_echelonize_init_opts(opts);
	opts->L = 1;

	double start_time = spasm_wtime();
	auto fact = spasm_echelonize(m2, opts);
	double end_time = spasm_wtime();
	std::cout << "echelonize time: " << end_time - start_time << "s\n";
	spasm_csr_free(m2);

	auto effrow = fact->p;
	auto rank = fact->U->n;
	auto pivots = fact->qinv;

	std::string outname = std::string(argv[2]) + ".Lp";
	FILE *out = fopen(outname.c_str(), "w");
	for (int i = 0; i < rank; i++){
		fprintf(out, "%d\n", effrow[i]);
	}
	fclose(out);

	outname = std::string(argv[2]) + ".Rq_1";
	out = fopen(outname.c_str(), "w");
	for (int i = 0; i < m; i++){
		fprintf(out, "%d\n", pivots[i]);
	}
	fclose(out);

	int *Rqinv = (int*)(spasm_malloc(m * sizeof(int)));
	struct spasm_csr *R = spasm_rref(fact, Rqinv);

	outname = std::string(argv[2]) + ".rref_" + std::to_string(p);
	out = fopen(outname.c_str(), "w");
	spasm_csr_save(R, out);
	fclose(out);

	outname = std::string(argv[2]) + ".Rq_rref";
	out = fopen(outname.c_str(), "w");
	for (int i = 0; i < m; i++){
		fprintf(out, "%d\n", Rqinv[i]);
	}
	fclose(out);

	spasm_csr_free(R);
	free(Rqinv);

	auto K = spasm_kernel(fact);
	fprintf(stderr, "Kernel basis matrix is %d x %d with %" PRId64 " nz\n", K->n, K->m, spasm_nnz(K));

	outname = std::string(argv[2]) + ".ker_" + std::to_string(p);
	out = fopen(outname.c_str(), "w");
	spasm_csr_save(K, out);
	fclose(out);
	spasm_csr_free(K);

	spasm_lu_free(fact);
	return 0;
}
