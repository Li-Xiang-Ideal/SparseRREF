(* ::Package:: *)

$SpaseRREFDirectory=DirectoryName[$InputFileName];
$SpaseRREFLibrary=With[{ext=Switch[$OperatingSystem,"Windows","dll","MacOSX","dylib","Unix","so"]},SortBy[FileNames[{$SpaseRREFDirectory<>"mathlink.*",If[FileNames["/.dockerenv"]=!={},"/usr/local/lib/mathlink.*",Nothing]}],If[StringExtract[#,"."->-1]===ext,0,1]&][[1]]]


Print["Sparse Reduced Row Echelon Form v0.3.5"]


modpmatmullib=LibraryFunctionLoad[$SpaseRREFLibrary,"modpmatmul",{{LibraryDataType[SparseArray],"Constant"},{LibraryDataType[SparseArray],"Constant"},{Integer},{Integer}},{LibraryDataType[SparseArray],Automatic}];
modrreflib=LibraryFunctionLoad[$SpaseRREFLibrary,"modrref",{{LibraryDataType[SparseArray],"Constant"},{Integer},{Integer},{Integer}},{LibraryDataType[SparseArray],Automatic}];
ratrreflib=LibraryFunctionLoad[$SpaseRREFLibrary,"rational_rref",{{LibraryDataType[ByteArray],"Constant"},{Integer},{Integer}},{LibraryDataType[ByteArray],Automatic}];
ratmatinvlib=LibraryFunctionLoad[$SpaseRREFLibrary,"ratmat_inv",{{LibraryDataType[ByteArray],"Constant"},{Integer}},{LibraryDataType[ByteArray],Automatic}];
modpmatmul[mat1_SparseArray,mat2_SparseArray,p_Integer,nthread_:1]:=modpmatmullib[mat1,mat2,p,nthread];
modprref[mat_SparseArray,p_Integer,method_:1,nthread_:1]:=With[{joinedmat=modrreflib[mat,p,method,nthread]},If[method=!=0,{joinedmat[[;;Length@mat]],Transpose[joinedmat[[Length@mat+1;;]]]},joinedmat]];
ratrref[mat_SparseArray,mode_:1,nthread_:1]:=BinaryDeserialize[ratrreflib[BinarySerialize[mat],mode,nthread]];
ratmatinv[mat_SparseArray,nthread_:1]:=BinaryDeserialize[ratmatinvlib[BinarySerialize[mat],nthread]];
