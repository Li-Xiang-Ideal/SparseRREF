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
  - "OutputMode" - return value of a function:
    0, "RREF": rref
    1, "RREF,Kernel": {rref, kernel}
    2, "RREF,Pivots": {rref, pivots} (only RationalRREF)
    3, "RREF,Kernel,Pivots": {rref, kernel, pivots} (only RationalRREF)
  - "Method":
    0, "RightAndLeft": right and left search
    1, "Right": only right search (chooses the leftmost independent columns as pivots)
    2, "Hybrid": hybrid
  - "Threads": number of threads (nonnegative integer).

  Example usage:
    Needs["SparseRREF`"];
    (* or: Needs["SparseRREF`", "/path/to/SparseRREF.m"]; *)
    
    mat = SparseArray @ { {1, 0, 2}, {1/2, 1/3, 1/4} };
    rref = RationalRREF[mat];
    {rref, kernel, pivots} = RationalRREF[mat, "OutputMode" -> "RREF,Kernel,Pivots", "Method" -> "Right", "Threads" -> $ProcessorCount];

    mat = SparseArray @ { {10, 0, 20}, {30, 40, 50} };
    p = 7;
    {rref, kernel} = ModRREF[mat, p, "OutputMode" -> "RREF,Kernel", "Method" -> "Hybrid", "Threads" -> 0];
*)

BeginPackage["SparseRREF`"];

Unprotect["SparseRREF`*"];

Options[RationalRREF] = Options[ModRREF] = {
  "OutputMode" -> "RREF",
  "Method" -> "RightAndLeft",
  "Threads" -> 1
};

RationalRREF::usage =
  "RationalRREF[mat, opts] computes the exact RREF of a sparse rational matrix. " <>
  "Default options: " <> ToString @ Options @ RationalRREF;
ModRREF::usage =
  "ModRREF[mat, p, opts] computes the exact RREF of a sparse integer matrix module prime p. " <>
  "Default options: " <> ToString @ Options @ ModRREF;

SyntaxInformation[RationalRREF] = {"ArgumentsPattern" -> {_, OptionsPattern[]}}
SyntaxInformation[ModRREF] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}}


Begin["`Private`"];

(* Load SparseRREF library *)

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

$modRREFLibFunction = 
  LibraryFunctionLoad[
    $sparseRREFLib,
    "modrref",
    {
      {LibraryDataType[SparseArray], "Constant"},
      {Integer},
      {Integer},
      {Integer},
      {Integer}
    },
    {LibraryDataType[SparseArray], Automatic}
  ];

(* Helper functions *)

option::value = "Invalid `1` option value: `2` -> `3`. Allowed values: `4`";

throwOptionError[f_, optionName_?StringQ, optionValue_, allowedValues_] := (
  Message[
    option::value,
    f,
    InputForm[optionName], 
    InputForm @ optionValue,
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
outputModeToInteger[RationalRREF] = <|
  0 -> 0,
  1 -> 1,
  2 -> 2,
  3 -> 3,
  "RREF" -> 0,
  "RREF,Kernel" -> 1,
  "RREF,Pivots" -> 2,
  "RREF,Kernel,Pivots" -> 3
|>;

(* TODO: modrref cannot return pivots, so only 0 and 1 are allowed. *)
outputModeToInteger[ModRREF] =
  Select[
    outputModeToInteger[RationalRREF], 
    MemberQ[{0, 1}, #] &
  ];

parseMethod[f_][method_] :=
  With[
    {$res = methodToInteger[method]},
    If[MissingQ[$res],
      throwOptionError[f, "Method", method, InputForm @ Keys @ methodToInteger],
      $res
    ]
  ];

parseOutputMode[f_][outputMode_] :=
  With[
    {$res = outputModeToInteger[f][outputMode]},
    If[MissingQ[$res],
      throwOptionError[f, "OutputMode", outputMode, InputForm @ Keys @ outputModeToInteger @ f],
      $res
    ]
  ];

parseThreads[f_][threads_] :=
  If[IntegerQ[threads] && threads >= 0,
    threads,
    throwOptionError[f, "Threads", threads, "0,1,2..." ]
  ];


(* Public functions *)

RationalRREF[mat_SparseArray, opts : OptionsPattern[] ] :=
  Catch @ With[
    {
      $outputMode = parseOutputMode[RationalRREF] @ OptionValue["OutputMode"],
      $method = parseMethod[RationalRREF] @ OptionValue["Method"],
      $threads = parseThreads[RationalRREF] @ OptionValue["Threads"]
    },
    BinaryDeserialize @ $rationalRREFLibFunction[
      BinarySerialize[mat],
      $outputMode,
      $method,
      $threads
    ]
  ];

ModRREF[mat_SparseArray, p_?IntegerQ, opts : OptionsPattern[] ] := 
  Catch @ With[
    {
      $outputMode = parseOutputMode[ModRREF] @ OptionValue["OutputMode"],
      $method = parseMethod[ModRREF] @ OptionValue["Method"],
      $threads = parseThreads[ModRREF] @ OptionValue["Threads"]
    },
    With[
      {
        joinedmat = $modRREFLibFunction[
          mat,
          p,
          $outputMode,
          $method,
          $threads
        ]
      },
      Switch[$outputMode,
        0,
        joinedmat,
        1,
        {
          joinedmat[[;; Length @ mat]],
          Transpose @ joinedmat[[Length @ mat + 1;;]]
        },
        (* TODO: support "OutputMode" -> 2 and 3 (pivots), similar to RationalRREF *)
        _,
        $Failed
      ]
    ]
  ];

With[{syms = Names["SparseRREF`*"]},
  SetAttributes[syms, {Protected, ReadProtected}]
];  

End[];

EndPackage[];