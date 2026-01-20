(* ::Package:: *)

(* 
  Mathematica interface for SparseRREF library.
  SparseRREF is a C++ library that computes exact RREF with row and column permutations of a sparse matrix over finite field or rational field.
  See details at https://github.com/munuxi/SparseRREF
  
  Prerequisites:
  - Compile mma_link.cpp to shared library mathlink.$EXT ($EXT = "dll" on Windows, "so" on Linux, "dylib" on macOS)
  - Store SparseRREF.wl in the same directory.

  Available functions:
  - SparseRREF

  Options:
  - "Modulus":
    0: compute RREF over rational field.
    prime number p: compute RREF over finite field Z/p (integers modulo p).
  - "OutputMode" - return value of a function:
    0, "RREF": rref
    1, "RREF,Kernel": {rref, kernel}
    2, "RREF,Pivots": {rref, pivots} (only for Modulus -> 0)
    3, "RREF,Kernel,Pivots": {rref, kernel, pivots} (only for Modulus -> 0)
  - "Method":
    0, "RightAndLeft": right and left search
    1, "Right": only right search (chooses the leftmost independent columns as pivots)
    2, "Hybrid": hybrid
  - "Threads": number of threads (nonnegative integer).

  Example usage:
    Needs["SparseRREF`"];
    (* or: Needs["SparseRREF`", "/path/to/SparseRREF.wl"]; *)
    
    mat = SparseArray @ { {1, 0, 2}, {1/2, 1/3, 1/4} };
    rref = SparseRREF[mat];
    {rref, kernel, pivots} = SparseRREF[mat, "OutputMode" -> "RREF,Kernel,Pivots", "Method" -> "Right", "Threads" -> $ProcessorCount];

    mat = SparseArray @ { {10, 0, 20}, {30, 40, 50} };
    p = 7;
    {rref, kernel} = SparseRREF[mat, Modulus -> p, "OutputMode" -> "RREF,Kernel", "Method" -> "Hybrid", "Threads" -> 0];
*)

BeginPackage["SparseRREF`"];

Unprotect["SparseRREF`*"];

Options[SparseRREF] = {
  Modulus -> 0,
  "OutputMode" -> "RREF",
  "Method" -> "RightAndLeft",
  "Threads" -> 1
};

SparseRREF::usage =
  "SparseRREF[mat, opts] computes the exact RREF of a sparse rational matrix " <>
  "or a sparse integer matrix modulo prime p (if Modulus -> p is specified). " <>
  "Default options: " <> ToString @ Options @ SparseRREF;

SyntaxInformation[SparseRREF] = {"ArgumentsPattern" -> {_, OptionsPattern[]}}

SparseRREF::findlib = "SparseRREF library \"`1`\" not found at `2`";
SparseRREF::optionvalue = "Invalid SparseRREF option value: `1` -> `2`. Allowed values: `3`";
SparseRREF::rettype = "SparseRREF should return SparseArray or List, but returned: `1`"

Begin["`Private`"];

(* Load SparseRREF library *)

$sparseRREFDirectory = DirectoryName[$InputFileName];
(* TODO rename e.g. to SparseRREF_MMA or SparseRREF_LibraryLink *)
$sparseRREFLibName = "mathlink";

(* TODO: shall we search in all directories from $LibraryPath? *)
$sparseRREFLib = FindLibrary @ FileNameJoin @ {$sparseRREFDirectory, $sparseRREFLibName};

If[FailureQ[$sparseRREFLib],
  Message[SparseRREF::findlib, $sparseRREFLibName, $sparseRREFDirectory];
];

(* TODO: provide API for other exported functions: modpmatmul, ratmat_inv *)

rationalRREFLibFunction =
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

modRREFLibFunction = 
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

parseThreads[threads_] :=
  If[IntegerQ[threads] && threads >= 0,
    threads,
    throwOptionError["Threads", threads, "0,1,2..." ]
  ];


checkResult[res_SparseArray] := res;
checkResult[res_List] := res;
checkResult[res_] :=
  (
    Message[SparseRREF::rettype, res];
    Throw[$Failed]
  );

(* Define public function SparseRREF[] *)

SparseRREF[mat_SparseArray, opts : OptionsPattern[] ] :=
  Catch @ With[
    {
      $modulus = parseModulus @ OptionValue["Modulus"],
      $outputMode = parseOutputMode @ OptionValue["OutputMode"],
      $method = parseMethod @ OptionValue["Method"],
      $threads = parseThreads @ OptionValue["Threads"]
    },
    checkResult @ If[$modulus == 0,
      rationalRREF[mat, $outputMode, $method, $threads],
      modRREF[mat, $modulus, $outputMode, $method, $threads]
    ]
  ];

rationalRREF[mat_SparseArray, outputMode_?IntegerQ, method_?IntegerQ, threads_?IntegerQ] :=
  BinaryDeserialize @ rationalRREFLibFunction[
    BinarySerialize[mat],
    outputMode,
    method,
    threads
  ];

modRREF[mat_SparseArray, p_?PrimeQ, outputMode_?IntegerQ, method_?IntegerQ, threads_?IntegerQ] :=
  With[
    {
      joinedmat = modRREFLibFunction[
        mat,
        p,
        outputMode,
        method,
        threads
      ]
    },
    If[!MatchQ[joinedmat, _SparseArray],
      (* SparseRREF will print error message and return $Failed *)
      Return[joinedmat];
    ];
    Switch[outputMode,
      0,
      joinedmat,
      1,
      {
        joinedmat[[;; Length @ mat]],
        Transpose @ joinedmat[[Length @ mat + 1;;]]
      },
      (* TODO: support "OutputMode" -> 2 and 3 (pivots), similar to rationalRREF *)
      _,
      throwOptionError[
        "OutputMode",
        outputMode,
        InputForm @ Keys @ Select[
          outputModeToInteger, 
          MemberQ[{0, 1}, #] &
        ]
      ]
    ]
  ];

With[{syms = Names["SparseRREF`*"]},
  SetAttributes[syms, {Protected, ReadProtected}]
];  

End[];

EndPackage[];