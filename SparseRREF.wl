(* ::Package:: *)

(* 
  Mathematica interface for SparseRREF library.
  SparseRREF is a C++ library that computes exact RREF with row and column permutations of a sparse matrix over finite field or rational field.
  See details at https://github.com/munuxi/SparseRREF
  
  Prerequisites:
  - Compile sprreflink.cpp to shared library sprreflink.$EXT ($EXT = "dll" on Windows, "so" on Linux, "dylib" on macOS)
  - Store SparseRREF.wl in the same directory.
  
  Available functions:
  --------------------
  SparseRREF[mat, opts]
    Computes Row Reduced Echelon Form (and optionally kernel/pivots).
    
    Options:
    - "Modulus":
      0: compute over rational field (default).
      prime p: compute over finite field Z/p.
    - "OutputMode":
      0, "RREF": returns rref (default).
      1, "RREF,Kernel": returns {rref, kernel}.
      2, "RREF,Pivots": returns {rref, pivots}.
      3, "RREF,Kernel,Pivots": returns {rref, kernel, pivots}.
    - "Method":
      0, "RightAndLeft": right and left search (default).
      1, "Right": only right search (chooses the leftmost independent columns as pivots).
      2, "Hybrid": hybrid.
    - "BackwardSubstitution":
      True: submatrix rref[[ pivots[[All,1]], pivots[[All,2]] ]] is an identity matrix (default).
      False: submatrix is upper triangular.
    - "Threads": number of threads (Integer >= 0, with 0 meaning automatic).
    - "Verbose": True | False.
    - "PrintStep": Integer (print progress every n steps).
    
  SparseMatMul[matA, matB, opts]
    Computes the matrix multiplication of two sparse matrices.
    
    Options:
    - "Modulus":
      0: compute over rational field (default).
      prime p: compute over finite field Z/p.
    - "Threads": number of threads (Integer >= 0, with 0 meaning automatic).
    
  SparseTensorContract[tensorA, tensorB, IndexPairs, opts]
    Computes the tensor contraction of two sparse tensors, with the idxA-th index of tensorA contracted with the idxB-th index of tensorB for each {idxA, idxB} in IndexPairs.
    
    Options:
    - "Modulus":
      0: compute over rational field (default).
      prime p: compute over finite field Z/p.
    - "Threads": number of threads (Integer >= 0, with 0 meaning automatic).
    
  SparseTensorDot[tensorA, tensorB, opts]
    Computes the dot product of two sparse tensors.
    
    Options:
    - "Modulus":
      0: compute over rational field (default).
      prime p: compute over finite field Z/p.
    - "Threads": number of threads (Integer >= 0, with 0 meaning automatic).
    
  Example usage:
  --------------
    Needs["SparseRREF`"];
    (* or: Needs["SparseRREF`", "/path/to/SparseRREF.wl"]; *)
    
    (* --- RREF Example --- *)    
    (* Rationals *)
    mat = SparseArray @ { {1, 0, 2}, {1/2, 1/3, 1/4} };
    rref = SparseRREF[mat];
    {rref, kernel, pivots} = SparseRREF[mat, "OutputMode" -> "RREF,Kernel,Pivots", "Method" -> "Right", "BackwardSubstitution" -> True, "Threads" -> $ProcessorCount, "Verbose" -> True, "PrintStep" -> 10];
    
    (* Finite Field *)
    mat = SparseArray @ { {10, 0, 20}, {30, 40, 50} };
    p = 7;
    {rref, kernel} = SparseRREF[mat, Modulus -> p, "OutputMode" -> "RREF,Kernel", "Method" -> "Hybrid", "Threads" -> 1];
    
    (* --- MatMul Example --- *)
    (* Rationals *)
    matA = SparseArray @ { {1, 0, 2}, {1/2, 1/3, 1/4} };
    matB = SparseArray @ { {0, 1}, {1, 0}, {1, 1} };
    matC = SparseMatMul[matA, matB, "Threads" -> $ProcessorCount];
    
    (* Finite Field *)
    matA = SparseArray @ { {10, 0, 20}, {30, 40, 50} };
    matB = SparseArray @ { {1, 2}, {3, 4}, {5, 6} };
    p = 11;
    matC = SparseMatMul[matA, matB, Modulus -> p, "Threads" -> $ProcessorCount];
*)


BeginPackage["SparseRREF`"];

Unprotect["SparseRREF`*"];


Options[SparseRREF] = {
  Modulus -> 0,
  "OutputMode" -> "RREF",
  "Method" -> "RightAndLeft",
  "BackwardSubstitution" -> True,
  "Threads" -> 1,
  "Verbose" -> False,
  "PrintStep" -> 100
};

SparseRREF::usage =
  "SparseRREF[mat, opts] computes the exact RREF of a sparse rational matrix " <>
  "or a sparse integer matrix modulo prime p (if Modulus -> p is specified). " <>
  "Default options: " <> ToString @ Options @ SparseRREF;

SyntaxInformation[SparseRREF] = {"ArgumentsPattern" -> {_, OptionsPattern[]}}

SparseRREF::findlib = "SparseRREF library \"`1`\" not found at `2`";
SparseRREF::optionvalue = "Invalid SparseRREF option value: `1` -> `2`. Allowed values: `3`";
SparseRREF::rettype = "SparseRREF should return SparseArray or List, but returned: `1`";


Options[SparseMatMul] = {
  Modulus -> 0,
  "Threads" -> 1
};

SparseMatMul::usage =
  "SparseMatMul[matA, matB, opts] computes the matrix multiplication of two sparse rational matrices " <>
  "or two sparse integer matrices modulo prime p (if Modulus -> p is specified). " <>
  "Default options: " <> ToString @ Options @ SparseMatMul;

SyntaxInformation[SparseMatMul] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}}

SparseMatMul::optionvalue = "Invalid SparseMatMul option value: `1` -> `2`. Allowed values: `3`";
SparseMatMul::rettype = "SparseMatMul should return SparseArray, but returned: `1`";
SparseMatMul::dims = "Matrices `1` and `2` have incompatible dimensions.";


Options[SparseTensorContract] = {
  Modulus -> 0,
  "Threads" -> 1
};

SparseTensorContract::usage =
  "SparseTensorContract[tensorA, tensorB, IndexPairs, opts] computes the tensor contraction of " <>
  "two sparse rational tensors or two sparse integer tensors modulo prime p (if Modulus -> p is specified), " <>
  "with the idxA-th index of tensorA contracted with the idxB-th index of tensorB for each {idxA, idxB} in IndexPairs. " <>
  "Default options: " <> ToString @ Options @ SparseTensorContract;

SyntaxInformation[SparseTensorContract] = {"ArgumentsPattern" -> {_, _, _, OptionsPattern[]}}

SparseTensorContract::optionvalue = "Invalid SparseTensorContract option value: `1` -> `2`. Allowed values: `3`";
SparseTensorContract::rettype = "SparseTensorContract should return SparseArray, but returned: `1`";
SparseTensorContract::dims = "Tensors `1` and `2` have incompatible dimensions for contraction.";
SparseTensorContract::indexpairs = "IndexPairs `1` is invalid.";


Options[SparseTensorDot] = Options[SparseTensorContract];

SparseTensorDot::usage =
  "SparseTensorDot[tensorA, tensorB, opts] computes the dot product of two sparse rational tensors " <>
  "or two sparse integer tensors modulo prime p (if Modulus -> p is specified). " <>
  "Default options: " <> ToString @ Options @ SparseTensorDot;

SyntaxInformation[SparseTensorDot] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}}

SparseTensorDot::optionvalue = "Invalid SparseTensorDot option value: `1` -> `2`. Allowed values: `3`";
SparseTensorDot::rettype = "SparseTensorDot should return SparseArray, but returned: `1`";
SparseTensorDot::dims = "Tensors `1` and `2` have incompatible dimensions for dot product.";


Begin["`Private`"];

(* Load SparseRREF library *)

$sparseRREFDirectory = DirectoryName[$InputFileName];
$sparseRREFLibName = "sprreflink";

(* TODO: shall we search in all directories from $LibraryPath? *)
$sparseRREFLib = FindLibrary @ FileNameJoin @ {$sparseRREFDirectory, $sparseRREFLibName};

If[FailureQ[$sparseRREFLib],
  Message[SparseRREF::findlib, $sparseRREFLibName, $sparseRREFDirectory];
];


(* TODO: provide API for other exported function(s): sprref_rat_matinv *)

ratRREFLibFunction =
  LibraryFunctionLoad[
    $sparseRREFLib,
    "sprref_rat_rref",
    {
      {LibraryDataType[ByteArray], "Constant"},
      Integer,
      Integer,
      True | False,
      Integer,
      True | False,
      Integer
    },
    {LibraryDataType[ByteArray], Automatic}
  ];

modRREFLibFunction =
  LibraryFunctionLoad[
    $sparseRREFLib,
    "sprref_mod_rref",
    {
      {LibraryDataType[SparseArray], "Constant"},
      Integer,
      Integer,
      Integer,
      True | False,
      Integer,
      True | False,
      Integer
    },
    {LibraryDataType[ByteArray], Automatic}
  ];


ratMatMulLibFunction =
  LibraryFunctionLoad[
    $sparseRREFLib,
    "sprref_rat_matmul",
    {
      {LibraryDataType[ByteArray], "Constant"},
      {LibraryDataType[ByteArray], "Constant"},
      Integer
    },
    {LibraryDataType[ByteArray], Automatic}
  ];

modMatMulLibFunction =
  LibraryFunctionLoad[
    $sparseRREFLib,
    "sprref_mod_matmul",
    {
      {LibraryDataType[SparseArray], "Constant"},
      {LibraryDataType[SparseArray], "Constant"},
      Integer,
      Integer
    },
    {LibraryDataType[SparseArray], Automatic}
  ];


ratTensorContractLibFunction =
  LibraryFunctionLoad[
    $sparseRREFLib,
    "sprref_rat_tensor_contract",
    {
      {LibraryDataType[ByteArray], "Constant"},
      {LibraryDataType[ByteArray], "Constant"},
      {LibraryDataType[List, Integer, 1], "Constant"},
      {LibraryDataType[List, Integer, 1], "Constant"},
      Integer
    },
    {LibraryDataType[ByteArray], Automatic}
  ];

modTensorContractLibFunction =
  LibraryFunctionLoad[
    $sparseRREFLib,
    "sprref_mod_tensor_contract",
    {
      {LibraryDataType[SparseArray], "Constant"},
      {LibraryDataType[SparseArray], "Constant"},
      Integer,
      {LibraryDataType[List, Integer, 1], "Constant"},
      {LibraryDataType[List, Integer, 1], "Constant"},
      Integer
    },
    {LibraryDataType[SparseArray], Automatic}
  ];


(* Helper functions: parse options, validate etc. *)

throwOptionError[optionName_?StringQ, optionValue_, allowedValues_] := (
  Message[
    SparseRREF::optionvalue,
    InputForm[optionName], 
    InputForm[optionValue],
    allowedValues
  ];
  Throw[$Failed];
);

methodToInteger = <|
  0 -> 0,
  1 -> 1,
  2 -> 2,
  "RightAndLeft" -> 0,
  "Right" -> 1,
  "Hybrid" -> 2
|>;

(* TODO: maybe allow arbitrary lists, e.g. {"Pivots", "RREF", "Kernel"}? *)
outputModeToInteger = <|
  0 -> 0,
  1 -> 1,
  2 -> 2,
  3 -> 3,
  "RREF" -> 0,
  "RREF,Kernel" -> 1,
  "RREF,Pivots" -> 2,
  "RREF,Kernel,Pivots" -> 3
|>;

parseModulus[0] := 0;
parseModulus[p_?PrimeQ] /; p > 0 := p;
parseModulus[p_] := throwOptionError["Modulus", p, "0 or prime number"];

parseMethod[method_] :=
  With[
    {$res = methodToInteger[method]},
    If[MissingQ[$res],
      throwOptionError["Method", method, InputForm @ Keys @ methodToInteger],
      $res
    ]
  ];

parseOutputMode[outputMode_] :=
  With[
    {$res = outputModeToInteger[outputMode]},
    If[MissingQ[$res],
      throwOptionError["OutputMode", outputMode, InputForm @ Keys @ outputModeToInteger],
      $res
    ]
  ];

parseBackwardSubstitution[b_?BooleanQ] := b;
parseBackwardSubstitution[b_] := throwOptionError["BackwardSubstitution", b, {True, False}];

parseThreads[threads_?IntegerQ] /; threads >= 0 := threads;
parseThreads[threads_] := throwOptionError["Threads", threads, "0,1,2..." ];

parseVerbose[b_?BooleanQ] := b;
parseVerbose[b_] := throwOptionError["Verbose", b, {True, False}];

parsePrintStep[ps_?IntegerQ] /; ps > 0 := ps;
parsePrintStep[ps_] := throwOptionError["PrintStep", ps, "1,2,3..."];

checkResult[msg_, res_, pattern_] :=
  If[MatchQ[res, pattern],
    res,
    Message[msg, res];
    Throw[$Failed]
  ];
SetAttributes[checkResult, HoldFirst];


(* Define public function SparseRREF[] *)

SparseRREF[mat_SparseArray, opts : OptionsPattern[] ] :=
  Catch @ With[
    {
      $modulus = parseModulus @ OptionValue["Modulus"],
      $outputMode = parseOutputMode @ OptionValue["OutputMode"],
      $method = parseMethod @ OptionValue["Method"],
      $backwardSubstitution = parseBackwardSubstitution @ OptionValue["BackwardSubstitution"],
      $threads = parseThreads @ OptionValue["Threads"],
      $verbose = parseVerbose @ OptionValue["Verbose"],
      $printStep = parsePrintStep @ OptionValue["PrintStep"]
    },
    checkResult[
      SparseRREF::rettype,
      If[$modulus == 0,
        ratRREF[mat, $outputMode, $method, $backwardSubstitution, $threads, $verbose, $printStep],
        modRREF[mat, $modulus, $outputMode, $method, $backwardSubstitution, $threads, $verbose, $printStep]
      ],
      _SparseArray | _List
    ]
  ];

ratRREF[
    mat_SparseArray,
    outputMode_?IntegerQ,
    method_?IntegerQ,
    backwardSubstitution_?BooleanQ,
    threads_?IntegerQ,
    verbose_?BooleanQ,
    printStep_?IntegerQ
  ] :=
  BinaryDeserialize @ ratRREFLibFunction[
    BinarySerialize[mat],
    outputMode,
    method,
    backwardSubstitution,
    threads,
    verbose,
    printStep
  ];

modRREF[
    mat_SparseArray,
    p_?PrimeQ,
    outputMode_?IntegerQ,
    method_?IntegerQ,
    backwardSubstitution_?BooleanQ,
    threads_?IntegerQ,
    verbose_?BooleanQ,
    printStep_?IntegerQ
  ] :=
  BinaryDeserialize @ modRREFLibFunction[
    mat,
    p,
    outputMode,
    method,
    backwardSubstitution,
    threads,
    verbose,
    printStep
  ];


(* Define public function SparseMatMul[] *)

SparseMatMul[matA_SparseArray, matB_SparseArray, opts : OptionsPattern[] ] :=
  Catch @ With[
    {
      $modulus = parseModulus @ OptionValue["Modulus"],
      $threads = parseThreads @ OptionValue["Threads"],
      $dimA = Dimensions[matA],
      $dimB = Dimensions[matB]
    },
    If[Last[$dimA] != First[$dimB],
      Message[SparseMatMul::dims, matA, matB];
      Throw[$Failed];
    ];
    checkResult[
      SparseMatMul::rettype,
      If[$modulus == 0,
        ratMatMul[matA, matB, $threads],
        modMatMul[matA, matB, $modulus, $threads]
      ],
      _SparseArray
    ]
  ];

ratMatMul[
    matA_SparseArray,
    matB_SparseArray,
    threads_?IntegerQ
  ] :=
  BinaryDeserialize @ ratMatMulLibFunction[
    BinarySerialize[matA],
    BinarySerialize[matB],
    threads
  ];

modMatMul[
    matA_SparseArray,
    matB_SparseArray,
    p_?PrimeQ,
    threads_?IntegerQ
  ] :=
  modMatMulLibFunction[
    matA,
    matB,
    p,
    threads
  ];


(* Define public function SparseTensorContract[] *)

SparseTensorContract[
    tensorA_SparseArray,
    tensorB_SparseArray,
    indexPairs : {{_Integer ..} ..},
    opts : OptionsPattern[]
  ] :=
  Catch @ With[
    {
      $modulus = parseModulus @ OptionValue["Modulus"],
      $threads = parseThreads @ OptionValue["Threads"],
      $dimA = Dimensions[tensorA],
      $dimB = Dimensions[tensorB],
      $idxA = indexPairs[[All, 1]],
      $idxB = indexPairs[[All, 2]]
    },
    If[
      Or[
        Max[$idxA] > Length[$dimA],
        Max[$idxB] > Length[$dimB],
        Min[$idxA] < 1,
        Min[$idxB] < 1,
        Length[$idxA] != Length[$idxB]
      ],
      Message[SparseTensorContract::indexpairs, indexPairs];
      Throw[$Failed];
    ];
    If[And @@ Thread[ $dimA[[ $idxA ]] == $dimB[[ $idxB ]] ] == False,
      Message[SparseTensorContract::dims, tensorA, tensorB];
      Throw[$Failed];
    ];
    checkResult[
      SparseTensorContract::rettype,
      If[$modulus == 0,
        ratTensorContract[tensorA, tensorB, $idxA, $idxB, $threads],
        modTensorContract[tensorA, tensorB, $modulus, $idxA, $idxB, $threads]
      ],
      _SparseArray
    ]
  ];

ratTensorContract[
    tensorA_SparseArray,
    tensorB_SparseArray,
    idxA : {_Integer ..},
    idxB : {_Integer ..},
    threads_?IntegerQ
  ] :=
  BinaryDeserialize @ ratTensorContractLibFunction[
    BinarySerialize[tensorA],
    BinarySerialize[tensorB],
    idxA,
    idxB,
    threads
  ];

modTensorContract[
    tensorA_SparseArray,
    tensorB_SparseArray,
    p_?PrimeQ,
    idxA : {_Integer ..},
    idxB : {_Integer ..},
    threads_?IntegerQ
  ] :=
  modTensorContractLibFunction[
    tensorA,
    tensorB,
    p,
    idxA,
    idxB,
    threads
  ];

SparseTensorDot[
    tensorA_SparseArray,
    tensorB_SparseArray,
    opts : OptionsPattern[]
  ] :=
  Catch @ With[
    {
      $modulus = parseModulus @ OptionValue["Modulus"],
      $threads = parseThreads @ OptionValue["Threads"],
      $dimA = Dimensions[tensorA],
      $dimB = Dimensions[tensorB]
    },
    If[Last[$dimA] != First[$dimB],
      Message[SparseTensorDot::dims, tensorA, tensorB];
      Throw[$Failed];
    ];
    checkResult[
      SparseTensorDot::rettype,
      If[$modulus == 0,
        ratTensorContract[tensorA, tensorB, {Length[$dimA]}, {1}, $threads],
        modTensorContract[tensorA, tensorB, $modulus, {Length[$dimA]}, {1}, $threads]
      ],
      _SparseArray
    ]
  ];


With[{syms = Names["SparseRREF`*"]},
  SetAttributes[syms, {Protected, ReadProtected}]
];  

End[];

EndPackage[];
