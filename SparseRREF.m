(* ::Package:: *)

$SpaseRREFDirectory=DirectoryName[$InputFileName];


modrreflib=LibraryFunctionLoad[$SpaseRREFDirectory<>"mathlink.dll","modrref",{{LibraryDataType[SparseArray],"Constant"},{Integer},{Integer},{Integer}},{LibraryDataType[SparseArray],Automatic}];
ratrreflib=LibraryFunctionLoad[$SpaseRREFDirectory<>"mathlink.dll","rational_rref",{{LibraryDataType[ByteArray],"Constant"},{Integer},{Integer}},{LibraryDataType[ByteArray],Automatic}];
modprref[mat_SparseArray,p_Integer,method_:1,nthread_:1]:=With[{joinedmat=modrreflib[mat,p,method,nthread]},If[method=!=0,{joinedmat[[;;Length@mat]],joinedmat[[Length@mat+1;;]]},joinedmat]];
ratrref[mat_SparseArray,mode_:1,nthread_:1]:=BinaryDeserialize[ratrreflib[BinarySerialize[mat],mode,nthread]];
