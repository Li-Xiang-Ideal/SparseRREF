(* ::Package:: *)

(* 
  Mathematica interface for SparseRREF library.
  SparseRREF is a C++ library that computes exact RREF with row and column permutations of a sparse matrix over finite field or rational field.
  See details at https://github.com/munuxi/SparseRREF
  
  Prerequisites:
  - Compile mma_link.cpp to shared library mathlink.$EXT ($EXT = "dll" on Windows, "so" on Linux, "dylib" on macOS)
  - Store SparseRREF.m in the same directory.

  Available functions:
  - RationalRREF

  Example usage:
    (* Needs["SparseRREF`"]; *)
    Needs["SparseRREF`", "/path/to/SparseRREF.m"];
    mat = SparseArray @ { {1, 0}, {1/2, 1/3} };
    rref = RationalRREF[mat];
    {rref, kernel, pivots} = RationalRREF[mat, OutputMode -> 3, Threads -> $ProcessorCount];
*)

BeginPackage["SparseRREF`"];


Options[RationalRREF] = {
  OutputMode -> 0,
  Threads -> 1
};

RationalRREF::usage =
  "RationalRREF[mat, opts] computes the exact RREF of a sparse rational matrix. " <>
  "Default options: " <> ToString @ Options @ RationalRREF;

SyntaxInformation[RationalRREF] = {"ArgumentsPattern" -> {_, OptionsPattern[]}}

(* TODO: use meaningful names instead of integers*)
OutputMode::usage =
  "Output mode for RationalRREF:
  0: rref
  1: {rref, kernel}
  2: {rref, pivots}
  3: {rref, kernel, pivots}";

Threads::usage = "Number of threads used by SparseRREF functions.";


Begin["`Private`"];

$sparseRREFDirectory = DirectoryName[$InputFileName];

$sparseRREFLib = FindLibrary @ FileNameJoin @ {$sparseRREFDirectory, "mathlink"};

(* TODO: error message if $sparseRREFLib == $Failed *)

(* TODO: load other mathlink functions: modpmatmul, modrref, ratmat_inv *)

$rationalRREFLibFunction =
  LibraryFunctionLoad[
    $sparseRREFLib,
    "rational_rref",
    {
      {LibraryDataType[ByteArray], "Constant"},
      Integer,
      Integer
    },
    {LibraryDataType[ByteArray], Automatic}
  ];

RationalRREF[mat_SparseArray, opts : OptionsPattern[] ] :=
  BinaryDeserialize @
    $rationalRREFLibFunction[
      BinarySerialize[mat],
      OptionValue[OutputMode],
      OptionValue[Threads]
    ];

End[];

Protect[RationalRREF, OutputMode, Threads];

EndPackage[];