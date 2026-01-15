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
  - ModRREF

  Options:
  - "OutputMode":
    0: rref
    1: {rref, kernel}
    2: {rref, pivots}
    3: {rref, kernel, pivots}
  - "Method":
    0: right and left search
    1: only right search (chooses the leftmost independent columns as pivots)
    2: hybrid
  - "Threads": number of threads (integer).

  ModRREF supports only "OutputMode" -> 0 and 1, and does not support "Method".


  Example usage:
    Needs["SparseRREF`"];
    (* or: Needs["SparseRREF`", "/path/to/SparseRREF.m"]; *)
    
    mat = SparseArray @ { {1, 0, 2}, {1/2, 1/3, 1/4} };
    rref = RationalRREF[mat];
    {rref, kernel, pivots} = RationalRREF[mat, "OutputMode" -> 3, "Method" -> 1, "Threads" -> $ProcessorCount];

    mat = SparseArray @ { {10, 0, 20}, {30, 40, 50} };
    p = 7;
    {rref, kernel} = ModRREF[mat, p, "OutputMode" -> 1, "Threads" -> $ProcessorCount];
*)

BeginPackage["SparseRREF`"];

Unprotect["SparseRREF`*"];

(* TODO: use meaningful names instead of integers for OutputMode and Method *)
Options[RationalRREF] = {
  "OutputMode" -> 0,
  "Method" -> 0,
  "Threads" -> 1
};
Options[ModRREF] = {
  "OutputMode" -> 0,
  "Threads" -> 1
}

RationalRREF::usage =
  "RationalRREF[mat, opts] computes the exact RREF of a sparse rational matrix. " <>
  "Default options: " <> ToString @ Options @ RationalRREF;
ModRREF::usage =
  "ModRREF[mat, p, opts] computes the exact RREF of a sparse integer matrix module prime p. " <>
  "Default options: " <> ToString @ Options @ ModRREF;

SyntaxInformation[RationalRREF] = {"ArgumentsPattern" -> {_, OptionsPattern[]}}
SyntaxInformation[ModRREF] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}}


Begin["`Private`"];

$sparseRREFDirectory = DirectoryName[$InputFileName];

$sparseRREFLib = FindLibrary @ FileNameJoin @ {$sparseRREFDirectory, "mathlink"};

(* TODO: error message if $sparseRREFLib == $Failed *)

(* TODO: load other exported functions: modpmatmul, ratmat_inv *)

$rationalRREFLibFunction =
  LibraryFunctionLoad[
    $sparseRREFLib,
    "rational_rref",
    {
      {LibraryDataType[ByteArray], "Constant"},
      Integer,
      Integer,
      Integer
    },
    {LibraryDataType[ByteArray], Automatic}
  ];

RationalRREF[mat_SparseArray, opts : OptionsPattern[] ] :=
  BinaryDeserialize @
    $rationalRREFLibFunction[
      BinarySerialize[mat],
      OptionValue["OutputMode"],
      OptionValue["Method"],
      OptionValue["Threads"]
    ];

$modRREFLibFunction = 
  LibraryFunctionLoad[
    $sparseRREFLib,
    "modrref",
    {
      {LibraryDataType[SparseArray], "Constant"},
      {Integer},
      {Integer},
      {Integer}
    },
    {LibraryDataType[SparseArray], Automatic}
  ];

ModRREF[mat_SparseArray, p_?IntegerQ, opts : OptionsPattern[] ] := 
  With[
    {
      joinedmat = $modRREFLibFunction[
        mat,
        p,
        OptionValue["OutputMode"],
        OptionValue["Threads"]
      ]
    },
    Switch[OptionValue["OutputMode"],
      0,
      joinedmat,
      1,
      {
        joinedmat[[;; Length @ mat]],
        Transpose[joinedmat[[Length @ mat + 1;;]]]
      },
      (* TODO: support "OutputMode" -> 2 and 3 (pivots), similar to RationalRREF *)
      _,
      $Failed
    ]
  ];

With[{syms = Names["SparseRREF`*"]},
  SetAttributes[syms, {Protected, ReadProtected}]
];  

End[];

EndPackage[];